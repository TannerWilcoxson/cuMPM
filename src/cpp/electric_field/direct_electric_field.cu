#include "direct_electric_field.h"
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
// CUDA Vector Type Traits
// ---------------------------------------------------------------------------
template <typename T>
struct VectorType;

template <>
struct VectorType<float> {
    using Vec2 = float2;
};

template <>
struct VectorType<double> {
    using Vec2 = double2;
};

// ---------------------------------------------------------------------------
// Device Data Precision Conversion Kernels
// ---------------------------------------------------------------------------
__global__ void cast_double_to_float_kernel(const double* __restrict__ src, float* __restrict__ dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

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

template <typename Real>
__global__ void direct_field_kernel(
    const Real* __restrict__ d_dipoles,
    const Real* __restrict__ d_x_part,
    const Real* __restrict__ d_y_part,
    const Real* __restrict__ d_z_part,
    const Real* __restrict__ d_x_target,
    const Real* __restrict__ d_y_target,
    const Real* __restrict__ d_z_target,
    const Real* __restrict__ d_self_coef_r,
    const Real* __restrict__ d_self_coef_i,
    const Real* __restrict__ d_radius,
    const int*  __restrict__ d_quad_map,
    Real*       d_E_point,
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
    __shared__ Real sh_dip_xr[DIRECT_TILE_SIZE];
    __shared__ Real sh_dip_xi[DIRECT_TILE_SIZE];
    __shared__ Real sh_dip_yr[DIRECT_TILE_SIZE];
    __shared__ Real sh_dip_yi[DIRECT_TILE_SIZE];
    __shared__ Real sh_dip_zr[DIRECT_TILE_SIZE];
    __shared__ Real sh_dip_zi[DIRECT_TILE_SIZE];

    // Quadrupole shared memory
    __shared__ Real sh_q0r[DIRECT_TILE_SIZE];
    __shared__ Real sh_q0i[DIRECT_TILE_SIZE];
    __shared__ Real sh_q1r[DIRECT_TILE_SIZE];
    __shared__ Real sh_q1i[DIRECT_TILE_SIZE];
    __shared__ Real sh_q2r[DIRECT_TILE_SIZE];
    __shared__ Real sh_q2i[DIRECT_TILE_SIZE];
    __shared__ Real sh_q3r[DIRECT_TILE_SIZE];
    __shared__ Real sh_q3i[DIRECT_TILE_SIZE];
    __shared__ Real sh_q4r[DIRECT_TILE_SIZE];
    __shared__ Real sh_q4i[DIRECT_TILE_SIZE];

    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    Real E_xr = Real(0.0), E_xi = Real(0.0);
    Real E_yr = Real(0.0), E_yi = Real(0.0);
    Real E_zr = Real(0.0), E_zi = Real(0.0);

    Real G_0r = Real(0.0), G_0i = Real(0.0);
    Real G_1r = Real(0.0), G_1i = Real(0.0);
    Real G_2r = Real(0.0), G_2i = Real(0.0);
    Real G_3r = Real(0.0), G_3i = Real(0.0);
    Real G_4r = Real(0.0), G_4i = Real(0.0);

    Real jx = (j < M) ? d_x_target[j] : Real(0.0);
    Real jy = (j < M) ? d_y_target[j] : Real(0.0);
    Real jz = (j < M) ? d_z_target[j] : Real(0.0);

    int q_target = -1;
    if (solve_quadrupoles && j < N) {
        q_target = d_quad_map[j];
    }

    using Vec2 = typename VectorType<Real>::Vec2;
    const Vec2* dips = reinterpret_cast<const Vec2*>(d_dipoles);
    const Real INV_4PI = Real(1.0) / (Real(4.0) * Real(3.14159265358979323846));
    const Real inv_4pi_scaled = INV_4PI * ((mode == FieldCalcMode::SOLVER_AX) ? Real(-1.0) : Real(1.0));

    for (int tile_start = 0; tile_start < N; tile_start += DIRECT_TILE_SIZE) {
        int src = tile_start + tid;

        if (src < N) {
            sh_x[tid] = d_x_part[src];
            sh_y[tid] = d_y_part[src];
            sh_z[tid] = d_z_part[src];
            sh_radius[tid] = d_radius ? d_radius[src] : Real(0.0);
            Vec2 px = dips[src * 3 + 0];
            Vec2 py = dips[src * 3 + 1];
            Vec2 pz = dips[src * 3 + 2];
            sh_dip_xr[tid] = px.x;  sh_dip_xi[tid] = px.y;
            sh_dip_yr[tid] = py.x;  sh_dip_yi[tid] = py.y;
            sh_dip_zr[tid] = pz.x;  sh_dip_zi[tid] = pz.y;

            if (solve_quadrupoles) {
                int q_src = d_quad_map[src];
                if (q_src >= 0) {
                    const Vec2* quads = reinterpret_cast<const Vec2*>(d_dipoles + N * 3 * 2);
                    Vec2 q0 = quads[q_src * 5 + 0];
                    Vec2 q1 = quads[q_src * 5 + 1];
                    Vec2 q2 = quads[q_src * 5 + 2];
                    Vec2 q3 = quads[q_src * 5 + 3];
                    Vec2 q4 = quads[q_src * 5 + 4];
                    sh_q0r[tid] = q0.x; sh_q0i[tid] = q0.y;
                    sh_q1r[tid] = q1.x; sh_q1i[tid] = q1.y;
                    sh_q2r[tid] = q2.x; sh_q2i[tid] = q2.y;
                    sh_q3r[tid] = q3.x; sh_q3i[tid] = q3.y;
                    sh_q4r[tid] = q4.x; sh_q4i[tid] = q4.y;
                } else {
                    sh_q0r[tid] = Real(0.0); sh_q0i[tid] = Real(0.0);
                    sh_q1r[tid] = Real(0.0); sh_q1i[tid] = Real(0.0);
                    sh_q2r[tid] = Real(0.0); sh_q2i[tid] = Real(0.0);
                    sh_q3r[tid] = Real(0.0); sh_q3i[tid] = Real(0.0);
                    sh_q4r[tid] = Real(0.0); sh_q4i[tid] = Real(0.0);
                }
            }
        } else {
            sh_x[tid] = Real(0.0); sh_y[tid] = Real(0.0); sh_z[tid] = Real(0.0);
            sh_radius[tid] = Real(0.0);
            sh_dip_xr[tid] = Real(0.0); sh_dip_xi[tid] = Real(0.0);
            sh_dip_yr[tid] = Real(0.0); sh_dip_yi[tid] = Real(0.0);
            sh_dip_zr[tid] = Real(0.0); sh_dip_zi[tid] = Real(0.0);
            if (solve_quadrupoles) {
                sh_q0r[tid] = Real(0.0); sh_q0i[tid] = Real(0.0);
                sh_q1r[tid] = Real(0.0); sh_q1i[tid] = Real(0.0);
                sh_q2r[tid] = Real(0.0); sh_q2i[tid] = Real(0.0);
                sh_q3r[tid] = Real(0.0); sh_q3i[tid] = Real(0.0);
                sh_q4r[tid] = Real(0.0); sh_q4i[tid] = Real(0.0);
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
                Real px_r = sh_dip_xr[k], px_i = sh_dip_xi[k];
                Real py_r = sh_dip_yr[k], py_i = sh_dip_yi[k];
                Real pz_r = sh_dip_zr[k], pz_i = sh_dip_zi[k];

                Real p_dot_rhat_r = px_r * runit_x + py_r * runit_y + pz_r * runit_z;
                Real p_dot_rhat_i = px_i * runit_x + py_i * runit_y + pz_i * runit_z;

                Real E_dip_xr = Real(3.0) * p_dot_rhat_r * runit_x - px_r;
                Real E_dip_xi = Real(3.0) * p_dot_rhat_i * runit_x - px_i;
                Real E_dip_yr = Real(3.0) * p_dot_rhat_r * runit_y - py_r;
                Real E_dip_yi = Real(3.0) * p_dot_rhat_i * runit_y - py_i;
                Real E_dip_zr = Real(3.0) * p_dot_rhat_r * runit_z - pz_r;
                Real E_dip_zi = Real(3.0) * p_dot_rhat_i * runit_z - pz_i;

                E_xr += inv_4pi_scaled * inv_r3 * E_dip_xr;
                E_xi += inv_4pi_scaled * inv_r3 * E_dip_xi;
                E_yr += inv_4pi_scaled * inv_r3 * E_dip_yr;
                E_yi += inv_4pi_scaled * inv_r3 * E_dip_yi;
                E_zr += inv_4pi_scaled * inv_r3 * E_dip_zr;
                E_zi += inv_4pi_scaled * inv_r3 * E_dip_zi;

                if (solve_quadrupoles) {
                    // Load Q_new and scale to Q_phys = S^-1 Q_new
                    Real q0r_p = -sh_q0r[k], q0i_p = -sh_q0i[k];
                    Real q1r_p = Real(-2.0) * sh_q1r[k], q1i_p = Real(-2.0) * sh_q1i[k];
                    Real q2r_p = Real(-2.0) * sh_q2r[k], q2i_p = Real(-2.0) * sh_q2i[k];
                    Real q3r_p = -sh_q3r[k], q3i_p = -sh_q3i[k];
                    Real q4r_p = Real(-2.0) * sh_q4r[k], q4i_p = Real(-2.0) * sh_q4i[k];
                    Real qzzr_p = -(q0r_p + q3r_p);
                    Real qzzi_p = -(q0i_p + q3i_p);

                    Real Q_rhat_xr_p = q0r_p * runit_x + q1r_p * runit_y + q2r_p * runit_z;
                    Real Q_rhat_xi_p = q0i_p * runit_x + q1i_p * runit_y + q2i_p * runit_z;
                    Real Q_rhat_yr_p = q1r_p * runit_x + q3r_p * runit_y + q4r_p * runit_z;
                    Real Q_rhat_yi_p = q1i_p * runit_x + q3i_p * runit_y + q4i_p * runit_z;
                    Real Q_rhat_zr_p = q2r_p * runit_x + q4r_p * runit_y + qzzr_p * runit_z;
                    Real Q_rhat_zi_p = q2i_p * runit_x + q4i_p * runit_y + qzzi_p * runit_z;

                    Real Q_rhatrhat_r_p = runit_x * Q_rhat_xr_p + runit_y * Q_rhat_yr_p + runit_z * Q_rhat_zr_p;
                    Real Q_rhatrhat_i_p = runit_x * Q_rhat_xi_p + runit_y * Q_rhat_yi_p + runit_z * Q_rhat_zi_p;

                    Real E_quad_xr = Real(2.5) * Q_rhatrhat_r_p * runit_x - Q_rhat_xr_p;
                    Real E_quad_xi = Real(2.5) * Q_rhatrhat_i_p * runit_x - Q_rhat_xi_p;
                    Real E_quad_yr = Real(2.5) * Q_rhatrhat_r_p * runit_y - Q_rhat_yr_p;
                    Real E_quad_yi = Real(2.5) * Q_rhatrhat_i_p * runit_y - Q_rhat_yi_p;
                    Real E_quad_zr = Real(2.5) * Q_rhatrhat_r_p * runit_z - Q_rhat_zr_p;
                    Real E_quad_zi = Real(2.5) * Q_rhatrhat_i_p * runit_z - Q_rhat_zi_p;

                    E_xr += Real(3.0) * inv_4pi_scaled * inv_r4 * E_quad_xr;
                    E_xi += Real(3.0) * inv_4pi_scaled * inv_r4 * E_quad_xi;
                    E_yr += Real(3.0) * inv_4pi_scaled * inv_r4 * E_quad_yr;
                    E_yi += Real(3.0) * inv_4pi_scaled * inv_r4 * E_quad_yi;
                    E_zr += Real(3.0) * inv_4pi_scaled * inv_r4 * E_quad_zr;
                    E_zi += Real(3.0) * inv_4pi_scaled * inv_r4 * E_quad_zi;

                    if (q_target >= 0) {
                        Real rr_std0 = runit_x * runit_x;
                        Real rr_std1 = runit_x * runit_y;
                        Real rr_std2 = runit_x * runit_z;
                        Real rr_std3 = runit_y * runit_y;
                        Real rr_std4 = runit_y * runit_z;

                        Real G_dip_0r = Real(-5.0) * p_dot_rhat_r * rr_std0 + Real(2.0) * px_r * runit_x + p_dot_rhat_r;
                        Real G_dip_0i = Real(-5.0) * p_dot_rhat_i * rr_std0 + Real(2.0) * px_i * runit_x + p_dot_rhat_i;
                        Real G_dip_1r = Real(-5.0) * p_dot_rhat_r * rr_std1 + (px_r * runit_y + py_r * runit_x);
                        Real G_dip_1i = Real(-5.0) * p_dot_rhat_i * rr_std1 + (px_i * runit_y + py_i * runit_x);
                        Real G_dip_2r = Real(-5.0) * p_dot_rhat_r * rr_std2 + (px_r * runit_z + pz_r * runit_x);
                        Real G_dip_2i = Real(-5.0) * p_dot_rhat_i * rr_std2 + (px_i * runit_z + pz_i * runit_x);
                        Real G_dip_3r = Real(-5.0) * p_dot_rhat_r * rr_std3 + Real(2.0) * py_r * runit_y + p_dot_rhat_r;
                        Real G_dip_3i = Real(-5.0) * p_dot_rhat_i * rr_std3 + Real(2.0) * py_i * runit_y + p_dot_rhat_i;
                        Real G_dip_4r = Real(-5.0) * p_dot_rhat_r * rr_std4 + (py_r * runit_z + pz_r * runit_y);
                        Real G_dip_4i = Real(-5.0) * p_dot_rhat_i * rr_std4 + (py_i * runit_z + pz_i * runit_y);

                        // Scale G_dip to G_dip_new = S * G_dip_old
                        G_0r += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_0r);
                        G_0i += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_0i);
                        G_1r += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_1r);
                        G_1i += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_1i);
                        G_2r += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_2r);
                        G_2i += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_2i);
                        G_3r += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_3r);
                        G_3i += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_3i);
                        G_4r += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_4r);
                        G_4i += Real(3.0) * inv_4pi_scaled * inv_r4 * (Real(-1.0) * G_dip_4i);

                        Real Q_rr_rr_Q_0r = Real(2.0) * Q_rhat_xr_p * runit_x;
                        Real Q_rr_rr_Q_0i = Real(2.0) * Q_rhat_xi_p * runit_x;
                        Real Q_rr_rr_Q_1r = Q_rhat_xr_p * runit_y + Q_rhat_yr_p * runit_x;
                        Real Q_rr_rr_Q_1i = Q_rhat_xi_p * runit_y + Q_rhat_yi_p * runit_x;
                        Real Q_rr_rr_Q_2r = Q_rhat_xr_p * runit_z + Q_rhat_zr_p * runit_x;
                        Real Q_rr_rr_Q_2i = Q_rhat_xi_p * runit_z + Q_rhat_zi_p * runit_x;
                        Real Q_rr_rr_Q_3r = Real(2.0) * Q_rhat_yr_p * runit_y;
                        Real Q_rr_rr_Q_3i = Real(2.0) * Q_rhat_yi_p * runit_y;
                        Real Q_rr_rr_Q_4r = Q_rhat_yr_p * runit_z + Q_rhat_zr_p * runit_y;
                        Real Q_rr_rr_Q_4i = Q_rhat_yi_p * runit_z + Q_rhat_zi_p * runit_y;

                        Real factor = inv_4pi_scaled * inv_r5;

                        // Calculate physical G_quad
                        Real G_q0r = Real(-1.5) * q0r_p - Real(7.5) * Q_rhatrhat_r_p - Real(15.0) * Q_rr_rr_Q_0r + Real(52.5) * rr_std0 * Q_rhatrhat_r_p;
                        Real G_q0i = Real(-1.5) * q0i_p - Real(7.5) * Q_rhatrhat_i_p - Real(15.0) * Q_rr_rr_Q_0i + Real(52.5) * rr_std0 * Q_rhatrhat_i_p;
                        Real G_q1r = Real(-1.5) * q1r_p - Real(15.0) * Q_rr_rr_Q_1r + Real(52.5) * rr_std1 * Q_rhatrhat_r_p;
                        Real G_q1i = Real(-1.5) * q1i_p - Real(15.0) * Q_rr_rr_Q_1i + Real(52.5) * rr_std1 * Q_rhatrhat_i_p;
                        Real G_q2r = Real(-1.5) * q2r_p - Real(15.0) * Q_rr_rr_Q_2r + Real(52.5) * rr_std2 * Q_rhatrhat_r_p;
                        Real G_q2i = Real(-1.5) * q2i_p - Real(15.0) * Q_rr_rr_Q_2i + Real(52.5) * rr_std2 * Q_rhatrhat_i_p;
                        Real G_q3r = Real(-1.5) * q3r_p - Real(7.5) * Q_rhatrhat_r_p - Real(15.0) * Q_rr_rr_Q_3r + Real(52.5) * rr_std3 * Q_rhatrhat_r_p;
                        Real G_q3i = Real(-1.5) * q3i_p - Real(7.5) * Q_rhatrhat_i_p - Real(15.0) * Q_rr_rr_Q_3i + Real(52.5) * rr_std3 * Q_rhatrhat_i_p;
                        Real G_q4r = Real(-1.5) * q4r_p - Real(15.0) * Q_rr_rr_Q_4r + Real(52.5) * rr_std4 * Q_rhatrhat_r_p;
                        Real G_q4i = Real(-1.5) * q4i_p - Real(15.0) * Q_rr_rr_Q_4i + Real(52.5) * rr_std4 * Q_rhatrhat_i_p;

                        // Scale G_quad to G_quad_new = S * G_quad_old
                        G_0r += factor * (Real(-1.0) * G_q0r);
                        G_0i += factor * (Real(-1.0) * G_q0i);
                        G_1r += factor * (Real(-1.0) * G_q1r);
                        G_1i += factor * (Real(-1.0) * G_q1i);
                        G_2r += factor * (Real(-1.0) * G_q2r);
                        G_2i += factor * (Real(-1.0) * G_q2i);
                        G_3r += factor * (Real(-1.0) * G_q3r);
                        G_3i += factor * (Real(-1.0) * G_q3i);
                        G_4r += factor * (Real(-1.0) * G_q4r);
                        G_4i += factor * (Real(-1.0) * G_q4i);
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
            Real sc_r = d_self_coef_r[j] + self_corr;
            Real sc_i = d_self_coef_i[j];
            Vec2 pj_x = dips[j * 3 + 0];
            Vec2 pj_y = dips[j * 3 + 1];
            Vec2 pj_z = dips[j * 3 + 2];

            E_xr += sc_r * pj_x.x - sc_i * pj_x.y;
            E_xi += sc_r * pj_x.y + sc_i * pj_x.x;
            E_yr += sc_r * pj_y.x - sc_i * pj_y.y;
            E_yi += sc_r * pj_y.y + sc_i * pj_y.x;
            E_zr += sc_r * pj_z.x - sc_i * pj_z.y;
            E_zi += sc_r * pj_z.y + sc_i * pj_z.x;

            if (solve_quadrupoles && q_target >= 0) {
                Real q_sc_r = Real(2.5) * d_self_coef_r[j] / (r_j * r_j) + Real(3.0) * self_corr / (r_j * r_j);
                Real q_sc_i = Real(2.5) * d_self_coef_i[j] / (r_j * r_j);

                const Vec2* d_quad_d2 = reinterpret_cast<const Vec2*>(d_dipoles + N * 3 * 2);
                Vec2 q0 = d_quad_d2[q_target * 5 + 0];
                Vec2 q1 = d_quad_d2[q_target * 5 + 1];
                Vec2 q2 = d_quad_d2[q_target * 5 + 2];
                Vec2 q3 = d_quad_d2[q_target * 5 + 3];
                Vec2 q4 = d_quad_d2[q_target * 5 + 4];

                G_0r += q_sc_r * q0.x - q_sc_i * q0.y;
                G_0i += q_sc_r * q0.y + q_sc_i * q0.x;
                G_1r += q_sc_r * q1.x - q_sc_i * q1.y;
                G_1i += q_sc_r * q1.y + q_sc_i * q1.x;
                G_2r += q_sc_r * q2.x - q_sc_i * q2.y;
                G_2i += q_sc_r * q2.y + q_sc_i * q2.x;
                G_3r += q_sc_r * q3.x - q_sc_i * q3.y;
                G_3i += q_sc_r * q3.y + q_sc_i * q3.x;
                G_4r += q_sc_r * q4.x - q_sc_i * q4.y;
                G_4i += q_sc_r * q4.y + q_sc_i * q4.x;
            }
        }

        Vec2* out   = reinterpret_cast<Vec2*>(d_E_point);
        out[j * 3 + 0] = {E_xr, E_xi};
        out[j * 3 + 1] = {E_yr, E_yi};
        out[j * 3 + 2] = {E_zr, E_zi};

        if (solve_quadrupoles && q_target >= 0) {
            Vec2* out_G = reinterpret_cast<Vec2*>(d_E_point + M * 3 * 2);
            out_G[q_target * 5 + 0] = {G_0r, G_0i};
            out_G[q_target * 5 + 1] = {G_1r, G_1i};
            out_G[q_target * 5 + 2] = {G_2r, G_2i};
            out_G[q_target * 5 + 3] = {G_3r, G_3i};
            out_G[q_target * 5 + 4] = {G_4r, G_4i};
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
    if (d_float_self_coef_r) { cudaFree(d_float_self_coef_r); d_float_self_coef_r = nullptr; }
    if (d_float_self_coef_i) { cudaFree(d_float_self_coef_i); d_float_self_coef_i = nullptr; }
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
        CUDA_CHECK(cudaMalloc(&d_float_dipoles, (num_p * 3 + n_quads * 5) * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_float_self_coef_r, num_p * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_float_self_coef_i, num_p * sizeof(float)));
    }

    size_t num_targets = num_fp > 0 ? num_fp : num_p;
    if (num_targets > 0) {
        if (mode == FieldCalcMode::INTERACTION_FIELD && num_fp > 0) {
            CUDA_CHECK(cudaMalloc(&d_float_x_field, num_fp * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_float_y_field, num_fp * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_float_z_field, num_fp * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&d_float_E_point, (num_targets * 3 + n_quads * 5) * 2 * sizeof(float)));
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
            if (d_self_coef_r && d_self_coef_i) {
                cast_double_to_float_kernel<<<gridP, blockSize>>>(d_self_coef_r, d_float_self_coef_r, num_particles);
                cast_double_to_float_kernel<<<gridP, blockSize>>>(d_self_coef_i, d_float_self_coef_i, num_particles);
            } else {
                CUDA_CHECK(cudaMemset(d_float_self_coef_r, 0, num_particles * sizeof(float)));
                CUDA_CHECK(cudaMemset(d_float_self_coef_i, 0, num_particles * sizeof(float)));
            }

            size_t num_dip_elements = (num_particles * 3 + num_quads * 5) * 2;
            int gridDip = static_cast<int>((num_dip_elements + blockSize - 1) / blockSize);
            cast_double_to_float_kernel<<<gridDip, blockSize>>>(d_dipoles, d_float_dipoles, num_dip_elements);
        }

        if (mode == FieldCalcMode::INTERACTION_FIELD && num_field_points > 0) {
            int gridF = static_cast<int>((num_field_points + blockSize - 1) / blockSize);
            cast_double_to_float_kernel<<<gridF, blockSize>>>(d_x_field, d_float_x_field, num_field_points);
            cast_double_to_float_kernel<<<gridF, blockSize>>>(d_y_field, d_float_y_field, num_field_points);
            cast_double_to_float_kernel<<<gridF, blockSize>>>(d_z_field, d_float_z_field, num_field_points);
        }

        if (mode == FieldCalcMode::INTERACTION_FIELD) {
            size_t E_point_size_complex = num_field_points * 3 + num_quads * 5;
            CUDA_CHECK(cudaMemset(d_float_E_point, 0, E_point_size_complex * 2 * sizeof(float)));

            int blocksPerGrid = static_cast<int>((num_field_points + threadsPerBlock - 1) / threadsPerBlock);
            direct_field_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
                d_float_dipoles,
                d_float_x_part, d_float_y_part, d_float_z_part,
                d_float_x_field, d_float_y_field, d_float_z_field,
                d_float_self_coef_r, d_float_self_coef_i,
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
            cast_float_to_double_kernel<<<gridOut, blockSize>>>(d_float_E_point, d_E_point, E_point_size_complex * 2);
        } else {
            size_t E_point_size_complex = num_particles * 3 + num_quads * 5;
            CUDA_CHECK(cudaMemset(d_float_E_point, 0, E_point_size_complex * 2 * sizeof(float)));

            int blocksPerGrid = static_cast<int>((num_particles + threadsPerBlock - 1) / threadsPerBlock);
            direct_field_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
                d_float_dipoles,
                d_float_x_part, d_float_y_part, d_float_z_part,
                d_float_x_part, d_float_y_part, d_float_z_part,
                d_float_self_coef_r, d_float_self_coef_i,
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
            cast_float_to_double_kernel<<<gridOut, blockSize>>>(d_float_E_point, d_E_point, E_point_size_complex * 2);
        }
    } else {
        // Double precision (FP64) execution path
        if (mode == FieldCalcMode::INTERACTION_FIELD) {
            size_t E_point_size_complex = num_field_points * 3 + num_quads * 5;
            CUDA_CHECK(cudaMemset(d_E_point, 0, E_point_size_complex * 2 * sizeof(double)));

            int blocksPerGrid = static_cast<int>((num_field_points + threadsPerBlock - 1) / threadsPerBlock);
            direct_field_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(
                d_dipoles,
                d_x_part, d_y_part, d_z_part,
                d_x_field, d_y_field, d_z_field,
                d_self_coef_r, d_self_coef_i,
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
            CUDA_CHECK(cudaMemset(d_E_point, 0, E_point_size_complex * 2 * sizeof(double)));

            int blocksPerGrid = static_cast<int>((num_particles + threadsPerBlock - 1) / threadsPerBlock);
            direct_field_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(
                d_dipoles,
                d_x_part, d_y_part, d_z_part,
                d_x_part, d_y_part, d_z_part,
                d_self_coef_r, d_self_coef_i,
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
