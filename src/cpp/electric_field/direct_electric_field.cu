#include "direct_electric_field.h"
#include "cuda_complex_ops.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(ans) { gpuAssert_direct((ans), __FILE__, __LINE__); }
inline void gpuAssert_direct(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " at " << file << ":" << line << "\n";
        if (abort) exit(code);
    }
}
#endif

#define DIRECT_TILE_SIZE 128

// ---------------------------------------------------------------------------
// Device Data Precision Conversion Kernels
// ---------------------------------------------------------------------------


__global__ void cast_float_to_double_kernel(const float* __restrict__ src, double* __restrict__ dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = static_cast<double>(src[idx]);
    }
}

// ---------------------------------------------------------------------------
// direct_field_kernel
//
// Each thread handles one TARGET particle j and accumulates contributions from
// all SOURCE particles i != j using the quasi-static free-space dipole/quadrupole tensors.
// Templated over generic floating point scalar type Real (float or double).
// ---------------------------------------------------------------------------

template <typename Real, typename Vec2 = typename Real2Traits<Real>::Vec2>
__global__ void direct_field_kernel(
    const Vec2* __restrict__ d_dipoles,
    const Real* __restrict__ d_x_part,
    const Real* __restrict__ d_y_part,
    const Real* __restrict__ d_z_part,
    const Real* __restrict__ d_x_target,
    const Real* __restrict__ d_y_target,
    const Real* __restrict__ d_z_target,
    const Vec2* __restrict__ d_self_coef,
    const Real* __restrict__ d_radius,
    const int*  __restrict__ d_quad_map,
    Vec2*       __restrict__ d_E_point,
    int         N,
    int         M,
    int         num_quads,
    bool        solve_quadrupoles,
    FieldCalcMode mode)
{
    __shared__ Real sh_x[DIRECT_TILE_SIZE];
    __shared__ Real sh_y[DIRECT_TILE_SIZE];
    __shared__ Real sh_z[DIRECT_TILE_SIZE];
    __shared__ Real sh_radius[DIRECT_TILE_SIZE];
    __shared__ Vec2 sh_dip_x[DIRECT_TILE_SIZE];
    __shared__ Vec2 sh_dip_y[DIRECT_TILE_SIZE];
    __shared__ Vec2 sh_dip_z[DIRECT_TILE_SIZE];

    // Quadrupole shared memory
    __shared__ Vec2 sh_q0[DIRECT_TILE_SIZE];
    __shared__ Vec2 sh_q1[DIRECT_TILE_SIZE];
    __shared__ Vec2 sh_q2[DIRECT_TILE_SIZE];
    __shared__ Vec2 sh_q3[DIRECT_TILE_SIZE];
    __shared__ Vec2 sh_q4[DIRECT_TILE_SIZE];

    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    Vec2 zero = Real2Traits<Real>::make(0.0, 0.0);

    Vec2 E_x = zero;
    Vec2 E_y = zero;
    Vec2 E_z = zero;

    Vec2 G_0 = zero;
    Vec2 G_1 = zero;
    Vec2 G_2 = zero;
    Vec2 G_3 = zero;
    Vec2 G_4 = zero;

    Real jx = (j < M) ? d_x_target[j] : Real(0.0);
    Real jy = (j < M) ? d_y_target[j] : Real(0.0);
    Real jz = (j < M) ? d_z_target[j] : Real(0.0);

    int q_target = -1;
    if (solve_quadrupoles && j < N) {
        q_target = d_quad_map[j];
    }

    const Real INV_4PI = Real(1.0) / (Real(4.0) * Real(3.14159265358979323846));
    const Real inv_4pi_scaled = INV_4PI * ((mode == FieldCalcMode::SOLVER_AX) ? Real(-1.0) : Real(1.0));

    for (int tile_start = 0; tile_start < N; tile_start += DIRECT_TILE_SIZE) {
        int src = tile_start + tid;

        if (src < N) {
            sh_x[tid] = d_x_part[src];
            sh_y[tid] = d_y_part[src];
            sh_z[tid] = d_z_part[src];
            sh_radius[tid] = d_radius ? d_radius[src] : Real(0.0);
            sh_dip_x[tid] = d_dipoles[src * 3 + 0];
            sh_dip_y[tid] = d_dipoles[src * 3 + 1];
            sh_dip_z[tid] = d_dipoles[src * 3 + 2];

            if (solve_quadrupoles) {
                int q_src = d_quad_map[src];
                if (q_src >= 0) {
                    const Vec2* quads = d_dipoles + N * 3;
                    sh_q0[tid] = quads[q_src * 5 + 0];
                    sh_q1[tid] = quads[q_src * 5 + 1];
                    sh_q2[tid] = quads[q_src * 5 + 2];
                    sh_q3[tid] = quads[q_src * 5 + 3];
                    sh_q4[tid] = quads[q_src * 5 + 4];
                } else {
                    sh_q0[tid] = zero;
                    sh_q1[tid] = zero;
                    sh_q2[tid] = zero;
                    sh_q3[tid] = zero;
                    sh_q4[tid] = zero;
                }
            }
        } else {
            sh_x[tid] = Real(0.0); sh_y[tid] = Real(0.0); sh_z[tid] = Real(0.0);
            sh_radius[tid] = Real(0.0);
            sh_dip_x[tid] = zero;
            sh_dip_y[tid] = zero;
            sh_dip_z[tid] = zero;
            if (solve_quadrupoles) {
                sh_q0[tid] = zero;
                sh_q1[tid] = zero;
                sh_q2[tid] = zero;
                sh_q3[tid] = zero;
                sh_q4[tid] = zero;
            }
        }
        __syncthreads();

        if (j < M) {
            int tile_end = min(DIRECT_TILE_SIZE, N - tile_start);
            for (int k = 0; k < tile_end; ++k) {
                int i = tile_start + k;
                
                // Exclude self-interaction for Solver
                if (mode == FieldCalcMode::SOLVER_AX && (i == j)) continue;

                // Source - Target
                Real rx = sh_x[k] - jx;
                Real ry = sh_y[k] - jy;
                Real rz = sh_z[k] - jz;
                Real r2 = rx*rx + ry*ry + rz*rz;

                // Soften/regularize field evaluation inside particle volume
                Real r_sub = sh_radius[k];
                Real min_dist = Real(1.0) * r_sub;
                Real min_dist2 = min_dist * min_dist;
                Real r_orig = sqrt(r2);
                Real eps_orig = (sizeof(Real) == 4) ? Real(1e-6) : Real(1e-9);
                if (r2 < min_dist2) {
                    if (r_orig > eps_orig) {
                        Real scale = sqrt(min_dist2 / r2);
                        rx *= scale;
                        ry *= scale;
                        rz *= scale;
                    }
                    r2 = min_dist2;
                }

                Real eps_sing = (sizeof(Real) == 4) ? Real(1e-12) : Real(1e-18);
                if (r2 < eps_sing) continue; // Skip exact coordinate overlaps to avoid singularity

                Real r  = sqrt(r2);

                Real inv_r = Real(1.0) / r;
                Real inv_r2 = Real(1.0) / r2;
                Real inv_r3 = inv_r2 * inv_r;
                Real inv_r4 = inv_r3 * inv_r;
                Real inv_r5 = inv_r4 * inv_r;

                Real runit_x = rx * inv_r;
                Real runit_y = ry * inv_r;
                Real runit_z = rz * inv_r;

                // 1. Dipole fields
                Vec2 px = sh_dip_x[k];
                Vec2 py = sh_dip_y[k];
                Vec2 pz = sh_dip_z[k];

                Vec2 p_dot_rhat = px * runit_x + py * runit_y + pz * runit_z;

                Vec2 E_dip_x = p_dot_rhat * (Real(3.0) * runit_x) - px;
                Vec2 E_dip_y = p_dot_rhat * (Real(3.0) * runit_y) - py;
                Vec2 E_dip_z = p_dot_rhat * (Real(3.0) * runit_z) - pz;

                Real dip_factor = inv_4pi_scaled * inv_r3;
                E_x += E_dip_x * dip_factor;
                E_y += E_dip_y * dip_factor;
                E_z += E_dip_z * dip_factor;

                if (solve_quadrupoles) {
                    // Load Q_new and scale to Q_phys = S^-1 Q_new
                    Vec2 q0_p = sh_q0[k] * Real(-1.0);
                    Vec2 q1_p = sh_q1[k] * Real(-2.0);
                    Vec2 q2_p = sh_q2[k] * Real(-2.0);
                    Vec2 q3_p = sh_q3[k] * Real(-1.0);
                    Vec2 q4_p = sh_q4[k] * Real(-2.0);
                    Vec2 qzz_p = (q0_p + q3_p) * Real(-1.0);

                    Vec2 Q_rhat_x_p = q0_p * runit_x + q1_p * runit_y + q2_p * runit_z;
                    Vec2 Q_rhat_y_p = q1_p * runit_x + q3_p * runit_y + q4_p * runit_z;
                    Vec2 Q_rhat_z_p = q2_p * runit_x + q4_p * runit_y + qzz_p * runit_z;

                    Vec2 Q_rhatrhat_p = Q_rhat_x_p * runit_x + Q_rhat_y_p * runit_y + Q_rhat_z_p * runit_z;

                    Vec2 E_quad_x = Q_rhatrhat_p * (Real(2.5) * runit_x) - Q_rhat_x_p;
                    Vec2 E_quad_y = Q_rhatrhat_p * (Real(2.5) * runit_y) - Q_rhat_y_p;
                    Vec2 E_quad_z = Q_rhatrhat_p * (Real(2.5) * runit_z) - Q_rhat_z_p;

                    Real quad_factor = Real(3.0) * inv_4pi_scaled * inv_r4;
                    E_x += E_quad_x * quad_factor;
                    E_y += E_quad_y * quad_factor;
                    E_z += E_quad_z * quad_factor;

                    if (q_target >= 0) {
                        Real rr_std0 = runit_x * runit_x;
                        Real rr_std1 = runit_x * runit_y;
                        Real rr_std2 = runit_x * runit_z;
                        Real rr_std3 = runit_y * runit_y;
                        Real rr_std4 = runit_y * runit_z;

                        Vec2 G_dip_0 = p_dot_rhat * (Real(-5.0) * rr_std0 + Real(1.0)) + px * (Real(2.0) * runit_x);
                        Vec2 G_dip_1 = p_dot_rhat * (Real(-5.0) * rr_std1) + (px * runit_y + py * runit_x);
                        Vec2 G_dip_2 = p_dot_rhat * (Real(-5.0) * rr_std2) + (px * runit_z + pz * runit_x);
                        Vec2 G_dip_3 = p_dot_rhat * (Real(-5.0) * rr_std3 + Real(1.0)) + py * (Real(2.0) * runit_y);
                        Vec2 G_dip_4 = p_dot_rhat * (Real(-5.0) * rr_std4) + (py * runit_z + pz * runit_y);

                        // Scale G_dip to G_dip_new = S * G_dip_old
                        Real g_dip_factor = Real(-3.0) * inv_4pi_scaled * inv_r4;
                        G_0 += G_dip_0 * g_dip_factor;
                        G_1 += G_dip_1 * g_dip_factor;
                        G_2 += G_dip_2 * g_dip_factor;
                        G_3 += G_dip_3 * g_dip_factor;
                        G_4 += G_dip_4 * g_dip_factor;

                        Vec2 Q_rr_rr_Q_0 = Q_rhat_x_p * (Real(2.0) * runit_x);
                        Vec2 Q_rr_rr_Q_1 = Q_rhat_x_p * runit_y + Q_rhat_y_p * runit_x;
                        Vec2 Q_rr_rr_Q_2 = Q_rhat_x_p * runit_z + Q_rhat_z_p * runit_x;
                        Vec2 Q_rr_rr_Q_3 = Q_rhat_y_p * (Real(2.0) * runit_y);
                        Vec2 Q_rr_rr_Q_4 = Q_rhat_y_p * runit_z + Q_rhat_z_p * runit_y;

                        Real factor = -inv_4pi_scaled * inv_r5;

                        // Calculate physical G_quad
                        Vec2 G_q0 = q0_p * Real(-1.5) + Q_rhatrhat_p * (Real(52.5) * rr_std0 - Real(7.5)) - Q_rr_rr_Q_0 * Real(15.0);
                        Vec2 G_q1 = q1_p * Real(-1.5) + Q_rhatrhat_p * (Real(52.5) * rr_std1) - Q_rr_rr_Q_1 * Real(15.0);
                        Vec2 G_q2 = q2_p * Real(-1.5) + Q_rhatrhat_p * (Real(52.5) * rr_std2) - Q_rr_rr_Q_2 * Real(15.0);
                        Vec2 G_q3 = q3_p * Real(-1.5) + Q_rhatrhat_p * (Real(52.5) * rr_std3 - Real(7.5)) - Q_rr_rr_Q_3 * Real(15.0);
                        Vec2 G_q4 = q4_p * Real(-1.5) + Q_rhatrhat_p * (Real(52.5) * rr_std4) - Q_rr_rr_Q_4 * Real(15.0);

                        G_0 += G_q0 * factor;
                        G_1 += G_q1 * factor;
                        G_2 += G_q2 * factor;
                        G_3 += G_q3 * factor;
                        G_4 += G_q4 * factor;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (j < M) {
        // Add self terms if SOLVER_AX
        if (mode == FieldCalcMode::SOLVER_AX) {
            Real r_j = d_radius[j];
            Real self_corr = INV_4PI / (r_j * r_j * r_j);
            Vec2 sc = d_self_coef[j];
            Vec2 self_factor = Real2Traits<Real>::make(static_cast<Real>(sc.x) + self_corr, static_cast<Real>(sc.y));

            Vec2 pj_x = d_dipoles[j * 3 + 0];
            Vec2 pj_y = d_dipoles[j * 3 + 1];
            Vec2 pj_z = d_dipoles[j * 3 + 2];

            E_x += self_factor * pj_x;
            E_y += self_factor * pj_y;
            E_z += self_factor * pj_z;

            if (solve_quadrupoles && q_target >= 0) {
                Real self_corr_Q = Real(3.0) * self_corr / (r_j * r_j);
                Real factor_sc = Real(2.5) / (r_j * r_j);
                Vec2 q_self_factor = Real2Traits<Real>::make(factor_sc * static_cast<Real>(sc.x) + self_corr_Q, factor_sc * static_cast<Real>(sc.y));

                const Vec2* d_quad_d2 = d_dipoles + N * 3;
                G_0 += q_self_factor * d_quad_d2[q_target * 5 + 0];
                G_1 += q_self_factor * d_quad_d2[q_target * 5 + 1];
                G_2 += q_self_factor * d_quad_d2[q_target * 5 + 2];
                G_3 += q_self_factor * d_quad_d2[q_target * 5 + 3];
                G_4 += q_self_factor * d_quad_d2[q_target * 5 + 4];
            }
        }

        d_E_point[j * 3 + 0] = E_x;
        d_E_point[j * 3 + 1] = E_y;
        d_E_point[j * 3 + 2] = E_z;

        if (solve_quadrupoles && q_target >= 0) {
            Vec2* out_G = d_E_point + M * 3;
            out_G[q_target * 5 + 0] = G_0;
            out_G[q_target * 5 + 1] = G_1;
            out_G[q_target * 5 + 2] = G_2;
            out_G[q_target * 5 + 3] = G_3;
            out_G[q_target * 5 + 4] = G_4;
        }
    }
}

// ---------------------------------------------------------------------------
// Direct_Electric_Field host implementation
// ---------------------------------------------------------------------------

bool Direct_Electric_Field::determinePrecisionMode(PrecisionMode precision) {
    if (precision == PrecisionMode::MIXED || precision == PrecisionMode::FP32) return true;
    if (precision == PrecisionMode::DOUBLE || precision == PrecisionMode::FP64) return false;

    // AUTO mode defaults to mixed precision (FP32)
    return true;
}

void Direct_Electric_Field::freeFloatBuffers() {
    if (d_float_x_part)      { cudaFree(d_float_x_part);      d_float_x_part      = nullptr; }
    if (d_float_y_part)      { cudaFree(d_float_y_part);      d_float_y_part      = nullptr; }
    if (d_float_z_part)      { cudaFree(d_float_z_part);      d_float_z_part      = nullptr; }
    if (d_float_radius)      { cudaFree(d_float_radius);      d_float_radius      = nullptr; }
    if (d_float_dipoles)     { cudaFree(d_float_dipoles);     d_float_dipoles     = nullptr; }
    if (d_float_self_coef)   { cudaFree(d_float_self_coef);   d_float_self_coef   = nullptr; }
    if (d_float_x_field)     { cudaFree(d_float_x_field);     d_float_x_field     = nullptr; }
    if (d_float_y_field)     { cudaFree(d_float_y_field);     d_float_y_field     = nullptr; }
    if (d_float_z_field)     { cudaFree(d_float_z_field);     d_float_z_field     = nullptr; }
    if (d_float_E_point)     { cudaFree(d_float_E_point);     d_float_E_point     = nullptr; }
    fp32_num_particles = 0;
    fp32_num_field_points = 0;
    fp32_num_quads = 0;
}

void Direct_Electric_Field::allocateFloatBuffers(size_t num_p, size_t num_fp, size_t n_quads) {
    if (num_p == fp32_num_particles && num_fp == fp32_num_field_points && n_quads == fp32_num_quads && d_float_x_part != nullptr) {
        return;
    }
    freeFloatBuffers();

    fp32_num_particles = num_p;
    fp32_num_field_points = num_fp;
    fp32_num_quads = n_quads;

    if (num_p > 0) {
        CUDA_CHECK(cudaMalloc(&d_float_x_part, num_p * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_float_y_part, num_p * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_float_z_part, num_p * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_float_radius, num_p * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_float_dipoles, (num_p * 3 + n_quads * 5) * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_float_self_coef, num_p * sizeof(float2)));
    }

    size_t num_targets = num_fp > 0 ? num_fp : num_p;
    if (num_targets > 0) {
        if (mode == FieldCalcMode::INTERACTION_FIELD && num_fp > 0) {
            CUDA_CHECK(cudaMalloc(&d_float_x_field, num_fp * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_float_y_field, num_fp * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_float_z_field, num_fp * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&d_float_E_point, (num_targets * 3 + n_quads * 5) * sizeof(float2)));
    }
}

Direct_Electric_Field::Direct_Electric_Field(const std::vector<double>& radius,
                                             FieldCalcMode mode,
                                             bool solve_quadrupoles,
                                             const std::vector<int>& quad_idxs,
                                             PrecisionMode precision)
    : Base_Electric_Field(mode, solve_quadrupoles, quad_idxs),
      host_radius(radius),
      precision_setting(precision)
{
    use_fp32 = determinePrecisionMode(precision);
}

Direct_Electric_Field::~Direct_Electric_Field() {
    if (d_radius) { cudaFree(d_radius); d_radius = nullptr; }
    freeFloatBuffers();
}

void Direct_Electric_Field::updateParticleCoordinates(
    const std::vector<double>& x_part,
    const std::vector<double>& y_part,
    const std::vector<double>& z_part)
{
    size_t N = x_part.size();
    if (N != num_particles) {
        if (d_radius) { cudaFree(d_radius); d_radius = nullptr; }
        if (N > 0 && !host_radius.empty()) {
            CUDA_CHECK(cudaMalloc(&d_radius, N * sizeof(double)));
        }
    }

    Base_Electric_Field::updateParticleCoordinates(x_part, y_part, z_part);

    if (N > 0 && !host_radius.empty()) {
        if (host_radius.size() != N) {
            throw std::invalid_argument("Direct_Electric_Field: host_radius size must match num_particles.");
        }
        CUDA_CHECK(cudaMemcpy(d_radius, host_radius.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void Direct_Electric_Field::electricField() {
    if (num_particles == 0) return;

    int threadsPerBlock = DIRECT_TILE_SIZE;

    if (use_fp32) {
        allocateFloatBuffers(num_particles, num_field_points, num_quads);

        int blockSize = 256;
        if (num_particles > 0) {
            int gridP = static_cast<int>((num_particles + blockSize - 1) / blockSize);
            cast_double_to_float_kernel<<<gridP, blockSize>>>(d_x_part, d_float_x_part, num_particles);
            cast_double_to_float_kernel<<<gridP, blockSize>>>(d_y_part, d_float_y_part, num_particles);
            cast_double_to_float_kernel<<<gridP, blockSize>>>(d_z_part, d_float_z_part, num_particles);
            if (d_radius) {
                cast_double_to_float_kernel<<<gridP, blockSize>>>(d_radius, d_float_radius, num_particles);
            } else {
                CUDA_CHECK(cudaMemset(d_float_radius, 0, num_particles * sizeof(float)));
            }
            if (d_self_coef) {
                size_t num_sc_elements = num_particles * 2;
                int gridSC = static_cast<int>((num_sc_elements + blockSize - 1) / blockSize);
                cast_double_to_float_kernel<<<gridSC, blockSize>>>(reinterpret_cast<const double*>(d_self_coef), reinterpret_cast<float*>(d_float_self_coef), num_sc_elements);
            } else {
                CUDA_CHECK(cudaMemset(d_float_self_coef, 0, num_particles * sizeof(float2)));
            }

            size_t num_dip_elements = (num_particles * 3 + num_quads * 5) * 2;
            int gridDip = static_cast<int>((num_dip_elements + blockSize - 1) / blockSize);
            cast_double_to_float_kernel<<<gridDip, blockSize>>>(reinterpret_cast<const double*>(d_dipoles), reinterpret_cast<float*>(d_float_dipoles), num_dip_elements);
        }

        if (mode == FieldCalcMode::INTERACTION_FIELD && num_field_points > 0) {
            int gridF = static_cast<int>((num_field_points + blockSize - 1) / blockSize);
            cast_double_to_float_kernel<<<gridF, blockSize>>>(d_x_field, d_float_x_field, num_field_points);
            cast_double_to_float_kernel<<<gridF, blockSize>>>(d_y_field, d_float_y_field, num_field_points);
            cast_double_to_float_kernel<<<gridF, blockSize>>>(d_z_field, d_float_z_field, num_field_points);
        }

        if (mode == FieldCalcMode::INTERACTION_FIELD) {
            size_t E_point_size_complex = num_field_points * 3 + num_quads * 5;
            CUDA_CHECK(cudaMemset(d_float_E_point, 0, E_point_size_complex * sizeof(float2)));

            int blocksPerGrid = static_cast<int>((num_field_points + threadsPerBlock - 1) / threadsPerBlock);
            direct_field_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
                d_float_dipoles,
                d_float_x_part, d_float_y_part, d_float_z_part,
                d_float_x_field, d_float_y_field, d_float_z_field,
                d_float_self_coef,
                d_float_radius,
                d_quad_map,
                d_float_E_point,
                static_cast<int>(num_particles),
                static_cast<int>(num_field_points),
                static_cast<int>(num_quads),
                solve_quadrupoles,
                FieldCalcMode::INTERACTION_FIELD
            );

            int gridOut = static_cast<int>((E_point_size_complex * 2 + blockSize - 1) / blockSize);
            cast_float_to_double_kernel<<<gridOut, blockSize>>>(reinterpret_cast<const float*>(d_float_E_point), reinterpret_cast<double*>(d_E_point), E_point_size_complex * 2);
        } else {
            size_t E_point_size_complex = num_particles * 3 + num_quads * 5;
            CUDA_CHECK(cudaMemset(d_float_E_point, 0, E_point_size_complex * sizeof(float2)));

            int blocksPerGrid = static_cast<int>((num_particles + threadsPerBlock - 1) / threadsPerBlock);
            direct_field_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
                d_float_dipoles,
                d_float_x_part, d_float_y_part, d_float_z_part,
                d_float_x_part, d_float_y_part, d_float_z_part,
                d_float_self_coef,
                d_float_radius,
                d_quad_map,
                d_float_E_point,
                static_cast<int>(num_particles),
                static_cast<int>(num_particles),
                static_cast<int>(num_quads),
                solve_quadrupoles,
                mode
            );

            int gridOut = static_cast<int>((E_point_size_complex * 2 + blockSize - 1) / blockSize);
            cast_float_to_double_kernel<<<gridOut, blockSize>>>(reinterpret_cast<const float*>(d_float_E_point), reinterpret_cast<double*>(d_E_point), E_point_size_complex * 2);
        }
    } else {
        // Double precision (FP64) execution path
        if (mode == FieldCalcMode::INTERACTION_FIELD) {
            size_t E_point_size_complex = num_field_points * 3 + num_quads * 5;
            CUDA_CHECK(cudaMemset(d_E_point, 0, E_point_size_complex * sizeof(double2)));

            int blocksPerGrid = static_cast<int>((num_field_points + threadsPerBlock - 1) / threadsPerBlock);
            direct_field_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(
                d_dipoles,
                d_x_part, d_y_part, d_z_part,
                d_x_field, d_y_field, d_z_field,
                d_self_coef,
                d_radius,
                d_quad_map,
                d_E_point,
                static_cast<int>(num_particles),
                static_cast<int>(num_field_points),
                static_cast<int>(num_quads),
                solve_quadrupoles,
                FieldCalcMode::INTERACTION_FIELD
            );
        } else {
            size_t E_point_size_complex = num_particles * 3 + num_quads * 5;
            CUDA_CHECK(cudaMemset(d_E_point, 0, E_point_size_complex * sizeof(double2)));

            int blocksPerGrid = static_cast<int>((num_particles + threadsPerBlock - 1) / threadsPerBlock);
            direct_field_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(
                d_dipoles,
                d_x_part, d_y_part, d_z_part,
                d_x_part, d_y_part, d_z_part,
                d_self_coef,
                d_radius,
                d_quad_map,
                d_E_point,
                static_cast<int>(num_particles),
                static_cast<int>(num_particles),
                static_cast<int>(num_quads),
                solve_quadrupoles,
                mode
            );
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Direct_Electric_Field::calculate() {
    particles_updated = false;
    field_points_updated = false;
    electricField();
}
