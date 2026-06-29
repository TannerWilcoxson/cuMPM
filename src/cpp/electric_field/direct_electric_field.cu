#include "direct_electric_field.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(ans) { gpuAssert_direct((ans), __FILE__, __LINE__); }
inline void gpuAssert_direct(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " at " << file << ":" << line << "\n";
        if (abort) exit(code);
    }
}

#define DIRECT_TILE_SIZE 128

// ---------------------------------------------------------------------------
// direct_field_kernel
//
// Each thread handles one TARGET particle j and accumulates contributions from
// all SOURCE particles i != j using the quasi-static free-space dipole/quadrupole tensors:
//
// Adds dipole and quadrupole fields to E_point, and gradients to E_point + N*3*2.

__global__ void direct_field_kernel(
    const double* __restrict__ d_dipoles,
    const double* __restrict__ d_x_part,
    const double* __restrict__ d_y_part,
    const double* __restrict__ d_z_part,
    const double* __restrict__ d_x_target,
    const double* __restrict__ d_y_target,
    const double* __restrict__ d_z_target,
    const double* __restrict__ d_self_coef_r,
    const double* __restrict__ d_self_coef_i,
    const double* __restrict__ d_radius,
    const int*    __restrict__ d_quad_map,
    double*       d_E_point,
    int           N,
    int           M,
    int           num_quads,
    bool          solve_quadrupoles,
    FieldCalcMode mode)
{
    __shared__ double sh_x[DIRECT_TILE_SIZE];
    __shared__ double sh_y[DIRECT_TILE_SIZE];
    __shared__ double sh_z[DIRECT_TILE_SIZE];
    __shared__ double sh_radius[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_xr[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_xi[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_yr[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_yi[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_zr[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_zi[DIRECT_TILE_SIZE];

    // Quadrupole shared memory
    __shared__ double sh_q0r[DIRECT_TILE_SIZE];
    __shared__ double sh_q0i[DIRECT_TILE_SIZE];
    __shared__ double sh_q1r[DIRECT_TILE_SIZE];
    __shared__ double sh_q1i[DIRECT_TILE_SIZE];
    __shared__ double sh_q2r[DIRECT_TILE_SIZE];
    __shared__ double sh_q2i[DIRECT_TILE_SIZE];
    __shared__ double sh_q3r[DIRECT_TILE_SIZE];
    __shared__ double sh_q3i[DIRECT_TILE_SIZE];
    __shared__ double sh_q4r[DIRECT_TILE_SIZE];
    __shared__ double sh_q4i[DIRECT_TILE_SIZE];

    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double E_xr = 0.0, E_xi = 0.0;
    double E_yr = 0.0, E_yi = 0.0;
    double E_zr = 0.0, E_zi = 0.0;

    double G_0r = 0.0, G_0i = 0.0;
    double G_1r = 0.0, G_1i = 0.0;
    double G_2r = 0.0, G_2i = 0.0;
    double G_3r = 0.0, G_3i = 0.0;
    double G_4r = 0.0, G_4i = 0.0;

    double jx = (j < M) ? d_x_target[j] : 0.0;
    double jy = (j < M) ? d_y_target[j] : 0.0;
    double jz = (j < M) ? d_z_target[j] : 0.0;

    int q_target = -1;
    if (solve_quadrupoles && j < N) {
        q_target = d_quad_map[j];
    }

    const double2* dips = reinterpret_cast<const double2*>(d_dipoles);
    const double INV_4PI = 1.0 / (4.0 * 3.14159265358979323846);
    const double inv_4pi_scaled = INV_4PI * ((mode == FieldCalcMode::SOLVER_AX) ? -1.0 : 1.0);

    for (int tile_start = 0; tile_start < N; tile_start += DIRECT_TILE_SIZE) {
        int src = tile_start + tid;

        if (src < N) {
            sh_x[tid] = d_x_part[src];
            sh_y[tid] = d_y_part[src];
            sh_z[tid] = d_z_part[src];
            sh_radius[tid] = d_radius ? d_radius[src] : 0.0;
            double2 px = dips[src * 3 + 0];
            double2 py = dips[src * 3 + 1];
            double2 pz = dips[src * 3 + 2];
            sh_dip_xr[tid] = px.x;  sh_dip_xi[tid] = px.y;
            sh_dip_yr[tid] = py.x;  sh_dip_yi[tid] = py.y;
            sh_dip_zr[tid] = pz.x;  sh_dip_zi[tid] = pz.y;

            if (solve_quadrupoles) {
                int q_src = d_quad_map[src];
                if (q_src >= 0) {
                    const double2* quads = reinterpret_cast<const double2*>(d_dipoles + N * 3 * 2);
                    double2 q0 = quads[q_src * 5 + 0];
                    double2 q1 = quads[q_src * 5 + 1];
                    double2 q2 = quads[q_src * 5 + 2];
                    double2 q3 = quads[q_src * 5 + 3];
                    double2 q4 = quads[q_src * 5 + 4];
                    sh_q0r[tid] = q0.x; sh_q0i[tid] = q0.y;
                    sh_q1r[tid] = q1.x; sh_q1i[tid] = q1.y;
                    sh_q2r[tid] = q2.x; sh_q2i[tid] = q2.y;
                    sh_q3r[tid] = q3.x; sh_q3i[tid] = q3.y;
                    sh_q4r[tid] = q4.x; sh_q4i[tid] = q4.y;
                } else {
                    sh_q0r[tid] = 0.0; sh_q0i[tid] = 0.0;
                    sh_q1r[tid] = 0.0; sh_q1i[tid] = 0.0;
                    sh_q2r[tid] = 0.0; sh_q2i[tid] = 0.0;
                    sh_q3r[tid] = 0.0; sh_q3i[tid] = 0.0;
                    sh_q4r[tid] = 0.0; sh_q4i[tid] = 0.0;
                }
            }
        } else {
            sh_x[tid] = 0.0; sh_y[tid] = 0.0; sh_z[tid] = 0.0;
            sh_radius[tid] = 0.0;
            sh_dip_xr[tid] = 0.0; sh_dip_xi[tid] = 0.0;
            sh_dip_yr[tid] = 0.0; sh_dip_yi[tid] = 0.0;
            sh_dip_zr[tid] = 0.0; sh_dip_zi[tid] = 0.0;
            if (solve_quadrupoles) {
                sh_q0r[tid] = 0.0; sh_q0i[tid] = 0.0;
                sh_q1r[tid] = 0.0; sh_q1i[tid] = 0.0;
                sh_q2r[tid] = 0.0; sh_q2i[tid] = 0.0;
                sh_q3r[tid] = 0.0; sh_q3i[tid] = 0.0;
                sh_q4r[tid] = 0.0; sh_q4i[tid] = 0.0;
            }
        }
        __syncthreads();

        if (j < M) {
            int tile_end = min(DIRECT_TILE_SIZE, N - tile_start);
            for (int k = 0; k < tile_end; ++k) {
                int i = tile_start + k;
                
                // Exclude self-interaction for Solver
                if (mode == FieldCalcMode::SOLVER_AX && (i == j)) continue;

                // Target - Source
                double rx = jx - sh_x[k];
                double ry = jy - sh_y[k];
                double rz = jz - sh_z[k];
                double r2 = rx*rx + ry*ry + rz*rz;

                // Soften/regularize field evaluation inside particle volume
                double r_sub = sh_radius[k];
                double min_dist = 1.0 * r_sub;
                double min_dist2 = min_dist * min_dist;
                double r_orig = sqrt(r2);
                if (r2 < min_dist2) {
                    if (r_orig > 1e-9) {
                        double scale = sqrt(min_dist2 / r2);
                        rx *= scale;
                        ry *= scale;
                        rz *= scale;
                    }
                    r2 = min_dist2;
                }

                if (r2 < 1e-18) continue; // Skip exact coordinate overlaps to avoid singularity

                double r  = sqrt(r2);

                double inv_r = 1.0 / r;
                double inv_r2 = 1.0 / r2;
                double inv_r3 = inv_r2 * inv_r;
                double inv_r4 = inv_r3 * inv_r;
                double inv_r5 = inv_r4 * inv_r;

                double runit_x = rx * inv_r;
                double runit_y = ry * inv_r;
                double runit_z = rz * inv_r;

                // 1. Dipole fields
                double px_r = sh_dip_xr[k], px_i = sh_dip_xi[k];
                double py_r = sh_dip_yr[k], py_i = sh_dip_yi[k];
                double pz_r = sh_dip_zr[k], pz_i = sh_dip_zi[k];

                double p_dot_rhat_r = px_r * runit_x + py_r * runit_y + pz_r * runit_z;
                double p_dot_rhat_i = px_i * runit_x + py_i * runit_y + pz_i * runit_z;

                double E_dip_xr = 3.0 * p_dot_rhat_r * runit_x - px_r;
                double E_dip_xi = 3.0 * p_dot_rhat_i * runit_x - px_i;
                double E_dip_yr = 3.0 * p_dot_rhat_r * runit_y - py_r;
                double E_dip_yi = 3.0 * p_dot_rhat_i * runit_y - py_i;
                double E_dip_zr = 3.0 * p_dot_rhat_r * runit_z - pz_r;
                double E_dip_zi = 3.0 * p_dot_rhat_i * runit_z - pz_i;

                E_xr += inv_4pi_scaled * inv_r3 * E_dip_xr;
                E_xi += inv_4pi_scaled * inv_r3 * E_dip_xi;
                E_yr += inv_4pi_scaled * inv_r3 * E_dip_yr;
                E_yi += inv_4pi_scaled * inv_r3 * E_dip_yi;
                E_zr += inv_4pi_scaled * inv_r3 * E_dip_zr;
                E_zi += inv_4pi_scaled * inv_r3 * E_dip_zi;

                if (solve_quadrupoles) {
                    // Load Q_new and scale to Q_phys = S^-1 Q_new
                    double q0r_p = -sh_q0r[k], q0i_p = -sh_q0i[k];
                    double q1r_p = -2.0 * sh_q1r[k], q1i_p = -2.0 * sh_q1i[k];
                    double q2r_p = -2.0 * sh_q2r[k], q2i_p = -2.0 * sh_q2i[k];
                    double q3r_p = -sh_q3r[k], q3i_p = -sh_q3i[k];
                    double q4r_p = -2.0 * sh_q4r[k], q4i_p = -2.0 * sh_q4i[k];
                    double qzzr_p = -(q0r_p + q3r_p);
                    double qzzi_p = -(q0i_p + q3i_p);

                    double Q_rhat_xr_p = q0r_p * runit_x + q1r_p * runit_y + q2r_p * runit_z;
                    double Q_rhat_xi_p = q0i_p * runit_x + q1i_p * runit_y + q2i_p * runit_z;
                    double Q_rhat_yr_p = q1r_p * runit_x + q3r_p * runit_y + q4r_p * runit_z;
                    double Q_rhat_yi_p = q1i_p * runit_x + q3i_p * runit_y + q4i_p * runit_z;
                    double Q_rhat_zr_p = q2r_p * runit_x + q4r_p * runit_y + qzzr_p * runit_z;
                    double Q_rhat_zi_p = q2i_p * runit_x + q4i_p * runit_y + qzzi_p * runit_z;

                    double Q_rhatrhat_r_p = runit_x * Q_rhat_xr_p + runit_y * Q_rhat_yr_p + runit_z * Q_rhat_zr_p;
                    double Q_rhatrhat_i_p = runit_x * Q_rhat_xi_p + runit_y * Q_rhat_yi_p + runit_z * Q_rhat_zi_p;

                    double E_quad_xr = 2.5 * Q_rhatrhat_r_p * runit_x - Q_rhat_xr_p;
                    double E_quad_xi = 2.5 * Q_rhatrhat_i_p * runit_x - Q_rhat_xi_p;
                    double E_quad_yr = 2.5 * Q_rhatrhat_r_p * runit_y - Q_rhat_yr_p;
                    double E_quad_yi = 2.5 * Q_rhatrhat_i_p * runit_y - Q_rhat_yi_p;
                    double E_quad_zr = 2.5 * Q_rhatrhat_r_p * runit_z - Q_rhat_zr_p;
                    double E_quad_zi = 2.5 * Q_rhatrhat_i_p * runit_z - Q_rhat_zi_p;

                    E_xr += 3.0 * inv_4pi_scaled * inv_r4 * E_quad_xr;
                    E_xi += 3.0 * inv_4pi_scaled * inv_r4 * E_quad_xi;
                    E_yr += 3.0 * inv_4pi_scaled * inv_r4 * E_quad_yr;
                    E_yi += 3.0 * inv_4pi_scaled * inv_r4 * E_quad_yi;
                    E_zr += 3.0 * inv_4pi_scaled * inv_r4 * E_quad_zr;
                    E_zi += 3.0 * inv_4pi_scaled * inv_r4 * E_quad_zi;

                    if (q_target >= 0) {
                        double rr0 = runit_x * runit_x - runit_z * runit_z;
                        double rr1 = 2.0 * runit_x * runit_y;
                        double rr2 = 2.0 * runit_x * runit_z;
                        double rr3 = runit_y * runit_y - runit_z * runit_z;
                        double rr4 = 2.0 * runit_y * runit_z;

                        double G_dip_0r = -5.0 * p_dot_rhat_r * rr0 + 2.0 * (px_r * runit_x - pz_r * runit_z) + p_dot_rhat_r;
                        double G_dip_0i = -5.0 * p_dot_rhat_i * rr0 + 2.0 * (px_i * runit_x - pz_i * runit_z) + p_dot_rhat_i;
                        double G_dip_1r = -5.0 * p_dot_rhat_r * rr1 + 2.0 * (px_r * runit_y + py_r * runit_x);
                        double G_dip_1i = -5.0 * p_dot_rhat_i * rr1 + 2.0 * (px_i * runit_y + py_i * runit_x);
                        double G_dip_2r = -5.0 * p_dot_rhat_r * rr2 + 2.0 * (px_r * runit_z + pz_r * runit_x);
                        double G_dip_2i = -5.0 * p_dot_rhat_i * rr2 + 2.0 * (px_i * runit_z + pz_i * runit_x);
                        double G_dip_3r = -5.0 * p_dot_rhat_r * rr3 + 2.0 * (py_r * runit_y - pz_r * runit_z) + p_dot_rhat_r;
                        double G_dip_3i = -5.0 * p_dot_rhat_i * rr3 + 2.0 * (py_i * runit_y - pz_i * runit_z) + p_dot_rhat_i;
                        double G_dip_4r = -5.0 * p_dot_rhat_r * rr4 + 2.0 * (py_r * runit_z + pz_r * runit_y);
                        double G_dip_4i = -5.0 * p_dot_rhat_i * rr4 + 2.0 * (py_i * runit_z + pz_i * runit_y);

                        // Scale G_dip to G_dip_new = S * G_dip_old
                        G_0r += 3.0 * inv_4pi_scaled * inv_r4 * (-1.0 * G_dip_0r);
                        G_0i += 3.0 * inv_4pi_scaled * inv_r4 * (-1.0 * G_dip_0i);
                        G_1r += 3.0 * inv_4pi_scaled * inv_r4 * (-0.5 * G_dip_1r);
                        G_1i += 3.0 * inv_4pi_scaled * inv_r4 * (-0.5 * G_dip_1i);
                        G_2r += 3.0 * inv_4pi_scaled * inv_r4 * (-0.5 * G_dip_2r);
                        G_2i += 3.0 * inv_4pi_scaled * inv_r4 * (-0.5 * G_dip_2i);
                        G_3r += 3.0 * inv_4pi_scaled * inv_r4 * (-1.0 * G_dip_3r);
                        G_3i += 3.0 * inv_4pi_scaled * inv_r4 * (-1.0 * G_dip_3i);
                        G_4r += 3.0 * inv_4pi_scaled * inv_r4 * (-0.5 * G_dip_4r);
                        G_4i += 3.0 * inv_4pi_scaled * inv_r4 * (-0.5 * G_dip_4i);

                        double Q_rr_rr_Q_0r = 2.0 * (Q_rhat_xr_p * runit_x - Q_rhat_zr_p * runit_z);
                        double Q_rr_rr_Q_0i = 2.0 * (Q_rhat_xi_p * runit_x - Q_rhat_zi_p * runit_z);
                        double Q_rr_rr_Q_1r = 2.0 * (Q_rhat_xr_p * runit_y + Q_rhat_yr_p * runit_x);
                        double Q_rr_rr_Q_1i = 2.0 * (Q_rhat_xi_p * runit_y + Q_rhat_yi_p * runit_x);
                        double Q_rr_rr_Q_2r = 2.0 * (Q_rhat_xr_p * runit_z + Q_rhat_zr_p * runit_x);
                        double Q_rr_rr_Q_2i = 2.0 * (Q_rhat_xi_p * runit_z + Q_rhat_zi_p * runit_x);
                        double Q_rr_rr_Q_3r = 2.0 * (Q_rhat_yr_p * runit_y - Q_rhat_zr_p * runit_z);
                        double Q_rr_rr_Q_3i = 2.0 * (Q_rhat_yi_p * runit_y - Q_rhat_zi_p * runit_z);
                        double Q_rr_rr_Q_4r = 2.0 * (Q_rhat_yr_p * runit_z + Q_rhat_zr_p * runit_y);
                        double Q_rr_rr_Q_4i = 2.0 * (Q_rhat_yi_p * runit_z + Q_rhat_zi_p * runit_y);

                        double factor = inv_4pi_scaled * inv_r5;

                        // Calculate physical G_quad
                        double G_q0r = -1.5 * q0r_p - 7.5 * Q_rhatrhat_r_p - 15.0 * Q_rr_rr_Q_0r + 52.5 * rr0 * Q_rhatrhat_r_p;
                        double G_q0i = -1.5 * q0i_p - 7.5 * Q_rhatrhat_i_p - 15.0 * Q_rr_rr_Q_0i + 52.5 * rr0 * Q_rhatrhat_i_p;
                        double G_q1r = -1.5 * q1r_p - 15.0 * Q_rr_rr_Q_1r + 52.5 * rr1 * Q_rhatrhat_r_p;
                        double G_q1i = -1.5 * q1i_p - 15.0 * Q_rr_rr_Q_1i + 52.5 * rr1 * Q_rhatrhat_i_p;
                        double G_q2r = -1.5 * q2r_p - 15.0 * Q_rr_rr_Q_2r + 52.5 * rr2 * Q_rhatrhat_r_p;
                        double G_q2i = -1.5 * q2i_p - 15.0 * Q_rr_rr_Q_2i + 52.5 * rr2 * Q_rhatrhat_i_p;
                        double G_q3r = -1.5 * q3r_p - 7.5 * Q_rhatrhat_r_p - 15.0 * Q_rr_rr_Q_3r + 52.5 * rr3 * Q_rhatrhat_r_p;
                        double G_q3i = -1.5 * q3i_p - 7.5 * Q_rhatrhat_i_p - 15.0 * Q_rr_rr_Q_3i + 52.5 * rr3 * Q_rhatrhat_i_p;
                        double G_q4r = -1.5 * q4r_p - 15.0 * Q_rr_rr_Q_4r + 52.5 * rr4 * Q_rhatrhat_r_p;
                        double G_q4i = -1.5 * q4i_p - 15.0 * Q_rr_rr_Q_4i + 52.5 * rr4 * Q_rhatrhat_i_p;

                        // Scale G_quad to G_quad_new = S * G_quad_old
                        G_0r += factor * (-1.0 * G_q0r);
                        G_0i += factor * (-1.0 * G_q0i);
                        G_1r += factor * (-0.5 * G_q1r);
                        G_1i += factor * (-0.5 * G_q1i);
                        G_2r += factor * (-0.5 * G_q2r);
                        G_2i += factor * (-0.5 * G_q2i);
                        G_3r += factor * (-1.0 * G_q3r);
                        G_3i += factor * (-1.0 * G_q3i);
                        G_4r += factor * (-0.5 * G_q4r);
                        G_4i += factor * (-0.5 * G_q4i);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (j < M) {
        // Add self terms if SOLVER_AX
        if (mode == FieldCalcMode::SOLVER_AX) {
            double r_j = d_radius[j];
            double self_corr = INV_4PI / (r_j * r_j * r_j);
            double sc_r = d_self_coef_r[j] + self_corr;
            double sc_i = d_self_coef_i[j];
            double2 pj_x = dips[j * 3 + 0];
            double2 pj_y = dips[j * 3 + 1];
            double2 pj_z = dips[j * 3 + 2];

            E_xr += sc_r * pj_x.x - sc_i * pj_x.y;
            E_xi += sc_r * pj_x.y + sc_i * pj_x.x;
            E_yr += sc_r * pj_y.x - sc_i * pj_y.y;
            E_yi += sc_r * pj_y.y + sc_i * pj_y.x;
            E_zr += sc_r * pj_z.x - sc_i * pj_z.y;
            E_zi += sc_r * pj_z.y + sc_i * pj_z.x;

            if (solve_quadrupoles && q_target >= 0) {
                double q_sc_r = 2.5 * d_self_coef_r[j] / (r_j * r_j) + 6.0 * self_corr / (r_j * r_j);
                double q_sc_i = 2.5 * d_self_coef_i[j] / (r_j * r_j);

                const double2* d_quad_d2 = reinterpret_cast<const double2*>(d_dipoles + N * 3 * 2);
                double2 q0 = d_quad_d2[q_target * 5 + 0];
                double2 q1 = d_quad_d2[q_target * 5 + 1];
                double2 q2 = d_quad_d2[q_target * 5 + 2];
                double2 q3 = d_quad_d2[q_target * 5 + 3];
                double2 q4 = d_quad_d2[q_target * 5 + 4];

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

        double2* out   = reinterpret_cast<double2*>(d_E_point);
        out[j * 3 + 0] = {E_xr, E_xi};
        out[j * 3 + 1] = {E_yr, E_yi};
        out[j * 3 + 2] = {E_zr, E_zi};

        if (solve_quadrupoles && q_target >= 0) {
            double2* out_G = reinterpret_cast<double2*>(d_E_point + M * 3 * 2);
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

Direct_Electric_Field::Direct_Electric_Field(const std::vector<double>& radius,
                                             FieldCalcMode mode,
                                             bool solve_quadrupoles,
                                             const std::vector<int>& quad_idxs)
    : host_radius(radius), mode(mode), solve_quadrupoles(solve_quadrupoles), quad_idxs(quad_idxs) {}

Direct_Electric_Field::~Direct_Electric_Field() {
    if (d_x_part)      { cudaFree(d_x_part);      d_x_part      = nullptr; }
    if (d_y_part)      { cudaFree(d_y_part);      d_y_part      = nullptr; }
    if (d_z_part)      { cudaFree(d_z_part);      d_z_part      = nullptr; }
    if (d_dipoles)     { cudaFree(d_dipoles);     d_dipoles     = nullptr; }
    if (d_E_point)     { cudaFree(d_E_point);     d_E_point     = nullptr; }
    if (d_self_coef_r) { cudaFree(d_self_coef_r); d_self_coef_r = nullptr; }
    if (d_self_coef_i) { cudaFree(d_self_coef_i); d_self_coef_i = nullptr; }
    if (d_radius)      { cudaFree(d_radius);      d_radius      = nullptr; }
    if (d_x_field)     { cudaFree(d_x_field);     d_x_field     = nullptr; }
    if (d_y_field)     { cudaFree(d_y_field);     d_y_field     = nullptr; }
    if (d_z_field)     { cudaFree(d_z_field);     d_z_field     = nullptr; }
    if (d_quad_idxs)   { cudaFree(d_quad_idxs);   d_quad_idxs   = nullptr; }
    if (d_quad_map)    { cudaFree(d_quad_map);    d_quad_map    = nullptr; }
}

void Direct_Electric_Field::updateParticleCoordinates(
    const std::vector<double>& x_part,
    const std::vector<double>& y_part,
    const std::vector<double>& z_part)
{
    if (x_part.size() != y_part.size() || x_part.size() != z_part.size()) {
        throw std::invalid_argument("Direct_Electric_Field: coordinate vectors must have equal size.");
    }
    size_t N = x_part.size();

    if (d_quad_idxs) { cudaFree(d_quad_idxs); d_quad_idxs = nullptr; }
    if (d_quad_map)  { cudaFree(d_quad_map);  d_quad_map  = nullptr; }

    if (solve_quadrupoles) {
        num_quads = quad_idxs.empty() ? N : quad_idxs.size();
        CUDA_CHECK(cudaMalloc(&d_quad_idxs, num_quads * sizeof(int)));
        std::vector<int> host_quad_idxs(num_quads);
        if (quad_idxs.empty()) {
            for (size_t i = 0; i < num_quads; ++i) {
                host_quad_idxs[i] = i;
            }
        } else {
            host_quad_idxs = quad_idxs;
        }
        CUDA_CHECK(cudaMemcpy(d_quad_idxs, host_quad_idxs.data(), num_quads * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_quad_map, N * sizeof(int)));
        std::vector<int> host_quad_map(N, -1);
        for (size_t q = 0; q < num_quads; ++q) {
            int p_idx = host_quad_idxs[q];
            if (p_idx >= 0 && p_idx < static_cast<int>(N)) {
                host_quad_map[p_idx] = q;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_quad_map, host_quad_map.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        num_quads = 0;
    }

    if (N != num_particles) {
        if (d_x_part)      { cudaFree(d_x_part);      d_x_part      = nullptr; }
        if (d_y_part)      { cudaFree(d_y_part);      d_y_part      = nullptr; }
        if (d_z_part)      { cudaFree(d_z_part);      d_z_part      = nullptr; }
        if (d_dipoles)     { cudaFree(d_dipoles);     d_dipoles     = nullptr; }
        if (d_self_coef_r) { cudaFree(d_self_coef_r); d_self_coef_r = nullptr; }
        if (d_self_coef_i) { cudaFree(d_self_coef_i); d_self_coef_i = nullptr; }
        if (d_radius)      { cudaFree(d_radius);      d_radius      = nullptr; }
        if (d_E_point) {
            cudaFree(d_E_point);
            d_E_point = nullptr;
        }

        num_particles = N;
        CUDA_CHECK(cudaMalloc(&d_x_part,      N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y_part,      N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z_part,      N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dipoles,     (N * 3 + num_quads * 5) * 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_r, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_radius,      N * sizeof(double)));

        CUDA_CHECK(cudaMemset(d_dipoles,     0, (N * 3 + num_quads * 5) * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_self_coef_r, 0, N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_self_coef_i, 0, N * sizeof(double)));

        if (num_field_points > 0) {
            CUDA_CHECK(cudaMalloc(&d_E_point, (num_field_points * 3 + num_quads * 5) * 2 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_E_point, 0, (num_field_points * 3 + num_quads * 5) * 2 * sizeof(double)));
        } else {
            CUDA_CHECK(cudaMalloc(&d_E_point, (N * 3 + num_quads * 5) * 2 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_E_point, 0, (N * 3 + num_quads * 5) * 2 * sizeof(double)));
        }

        if (host_radius.empty()) {
            host_radius.resize(N, 1.0);
        } else if (host_radius.size() != N) {
            throw std::invalid_argument("Direct_Electric_Field: radius size must equal num_particles.");
        }
        CUDA_CHECK(cudaMemcpy(d_radius, host_radius.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpy(d_x_part, x_part.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_part, y_part.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_part, z_part.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    particles_updated = true;
}

void Direct_Electric_Field::updateFieldCoordinates(
    const std::vector<double>& x_field,
    const std::vector<double>& y_field,
    const std::vector<double>& z_field)
{
    if (mode == FieldCalcMode::SOLVER_AX) {
        throw std::runtime_error("Direct_Electric_Field: Cannot update field coordinates when SOLVER_AX mode is active.");
    }
    if (x_field.size() != y_field.size() || x_field.size() != z_field.size()) {
        throw std::invalid_argument("Direct_Electric_Field: field coordinate vectors must have equal size.");
    }
    size_t M = x_field.size();
    if (M == 0) {
        if (d_x_field)     { cudaFree(d_x_field);     d_x_field     = nullptr; }
        if (d_y_field)     { cudaFree(d_y_field);     d_y_field     = nullptr; }
        if (d_z_field)     { cudaFree(d_z_field);     d_z_field     = nullptr; }
        if (d_E_point)     { cudaFree(d_E_point);     d_E_point     = nullptr; }
        num_field_points = 0;
        field_points_updated = false;
        
        if (num_particles > 0) {
            CUDA_CHECK(cudaMalloc(&d_E_point, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_E_point, 0, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double)));
        }
        return;
    }

    if (M != num_field_points) {
        if (d_x_field) { cudaFree(d_x_field); d_x_field = nullptr; }
        if (d_y_field) { cudaFree(d_y_field); d_y_field = nullptr; }
        if (d_z_field) { cudaFree(d_z_field); d_z_field = nullptr; }
        if (d_E_point) { cudaFree(d_E_point); d_E_point = nullptr; }

        num_field_points = M;
        CUDA_CHECK(cudaMalloc(&d_x_field, M * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y_field, M * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z_field, M * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_E_point, (M * 3 + num_quads * 5) * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_E_point, 0, (M * 3 + num_quads * 5) * 2 * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(d_x_field, x_field.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_field, y_field.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_field, z_field.data(), M * sizeof(double), cudaMemcpyHostToDevice));

    field_points_updated = true;
}

void Direct_Electric_Field::updateDipoles(const std::vector<double>& dip_x,
                                         const std::vector<double>& dip_y,
                                         const std::vector<double>& dip_z)
{
    if (dip_x.size() != num_particles || dip_y.size() != num_particles || dip_z.size() != num_particles) {
        throw std::invalid_argument("Direct_Electric_Field: input dipole vectors must match the allocated number of particles.");
    }
    if (num_particles == 0) return;

    std::vector<double> host_dips(num_particles * 3 * 2, 0.0);
    for (size_t p = 0; p < num_particles; ++p) {
        host_dips[(p * 3 + 0) * 2 + 0] = dip_x[p];
        host_dips[(p * 3 + 1) * 2 + 0] = dip_y[p];
        host_dips[(p * 3 + 2) * 2 + 0] = dip_z[p];
    }

    size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dips.data(), size_dipoles_bytes, cudaMemcpyHostToDevice));

    dipoles_updated = true;
}

void Direct_Electric_Field::updateDipolesComplex(
    const std::vector<double>& dip_xr, const std::vector<double>& dip_xi,
    const std::vector<double>& dip_yr, const std::vector<double>& dip_yi,
    const std::vector<double>& dip_zr, const std::vector<double>& dip_zi)
{
    if (dip_xr.size() != num_particles || dip_xi.size() != num_particles ||
        dip_yr.size() != num_particles || dip_yi.size() != num_particles ||
        dip_zr.size() != num_particles || dip_zi.size() != num_particles) {
        throw std::invalid_argument("Direct_Electric_Field: complex dipole component vectors must match the allocated number of particles.");
    }
    if (num_particles == 0) return;

    std::vector<double> host_dips(num_particles * 3 * 2);
    for (size_t p = 0; p < num_particles; ++p) {
        host_dips[(p * 3 + 0) * 2 + 0] = dip_xr[p];
        host_dips[(p * 3 + 0) * 2 + 1] = dip_xi[p];
        host_dips[(p * 3 + 1) * 2 + 0] = dip_yr[p];
        host_dips[(p * 3 + 1) * 2 + 1] = dip_yi[p];
        host_dips[(p * 3 + 2) * 2 + 0] = dip_zr[p];
        host_dips[(p * 3 + 2) * 2 + 1] = dip_zi[p];
    }

    size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dips.data(), size_dipoles_bytes, cudaMemcpyHostToDevice));

    dipoles_updated = true;
}

void Direct_Electric_Field::updateQuadrupoles(
    const std::vector<double>& quad_1, const std::vector<double>& quad_2,
    const std::vector<double>& quad_3, const std::vector<double>& quad_4,
    const std::vector<double>& quad_5)
{
    if (!solve_quadrupoles) return;
    if (quad_1.size() != num_quads || quad_2.size() != num_quads ||
        quad_3.size() != num_quads || quad_4.size() != num_quads ||
        quad_5.size() != num_quads) {
        throw std::invalid_argument("Direct_Electric_Field: quadrupole component vectors must match num_quads.");
    }
    if (num_quads == 0) return;

    std::vector<double> host_quads(num_quads * 5 * 2, 0.0);
    for (size_t q = 0; q < num_quads; ++q) {
        host_quads[(q * 5 + 0) * 2 + 0] = quad_1[q];
        host_quads[(q * 5 + 1) * 2 + 0] = quad_2[q];
        host_quads[(q * 5 + 2) * 2 + 0] = quad_3[q];
        host_quads[(q * 5 + 3) * 2 + 0] = quad_4[q];
        host_quads[(q * 5 + 4) * 2 + 0] = quad_5[q];
    }

    size_t offset = num_particles * 3 * 2;
    size_t size_quads_bytes = num_quads * 5 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_dipoles + offset, host_quads.data(), size_quads_bytes, cudaMemcpyHostToDevice));

    dipoles_updated = true;
}

void Direct_Electric_Field::updateQuadrupolesComplex(
    const std::vector<double>& quad_1r, const std::vector<double>& quad_1i,
    const std::vector<double>& quad_2r, const std::vector<double>& quad_2i,
    const std::vector<double>& quad_3r, const std::vector<double>& quad_3i,
    const std::vector<double>& quad_4r, const std::vector<double>& quad_4i,
    const std::vector<double>& quad_5r, const std::vector<double>& quad_5i)
{
    if (!solve_quadrupoles) return;
    if (quad_1r.size() != num_quads || quad_1i.size() != num_quads ||
        quad_2r.size() != num_quads || quad_2i.size() != num_quads ||
        quad_3r.size() != num_quads || quad_3i.size() != num_quads ||
        quad_4r.size() != num_quads || quad_4i.size() != num_quads ||
        quad_5r.size() != num_quads || quad_5i.size() != num_quads) {
        throw std::invalid_argument("Direct_Electric_Field: complex quadrupole component vectors must match num_quads.");
    }
    if (num_quads == 0) return;

    std::vector<double> host_quads(num_quads * 5 * 2, 0.0);
    for (size_t q = 0; q < num_quads; ++q) {
        host_quads[(q * 5 + 0) * 2 + 0] = quad_1r[q];
        host_quads[(q * 5 + 0) * 2 + 1] = quad_1i[q];
        host_quads[(q * 5 + 1) * 2 + 0] = quad_2r[q];
        host_quads[(q * 5 + 1) * 2 + 1] = quad_2i[q];
        host_quads[(q * 5 + 2) * 2 + 0] = quad_3r[q];
        host_quads[(q * 5 + 2) * 2 + 1] = quad_3i[q];
        host_quads[(q * 5 + 3) * 2 + 0] = quad_4r[q];
        host_quads[(q * 5 + 3) * 2 + 1] = quad_4i[q];
        host_quads[(q * 5 + 4) * 2 + 0] = quad_5r[q];
        host_quads[(q * 5 + 4) * 2 + 1] = quad_5i[q];
    }

    size_t offset = num_particles * 3 * 2;
    size_t size_quads_bytes = num_quads * 5 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_dipoles + offset, host_quads.data(), size_quads_bytes, cudaMemcpyHostToDevice));

    dipoles_updated = true;
}

void Direct_Electric_Field::getDipolesHost(std::vector<double>& host_dip_x,
                                          std::vector<double>& host_dip_y,
                                          std::vector<double>& host_dip_z) const
{
    if (num_particles == 0 || d_dipoles == nullptr) {
        throw std::runtime_error("Direct_Electric_Field: dipoles have not been allocated on GPU yet.");
    }

    host_dip_x.resize(num_particles);
    host_dip_y.resize(num_particles);
    host_dip_z.resize(num_particles);

    std::vector<double> host_dips(num_particles * 3 * 2);
    size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(host_dips.data(), d_dipoles, size_dipoles_bytes, cudaMemcpyDeviceToHost));

    for (size_t p = 0; p < num_particles; ++p) {
        host_dip_x[p] = host_dips[(p * 3 + 0) * 2 + 0];
        host_dip_y[p] = host_dips[(p * 3 + 1) * 2 + 0];
        host_dip_z[p] = host_dips[(p * 3 + 2) * 2 + 0];
    }
}

void Direct_Electric_Field::getDipolesComplexHost(
    std::vector<double>& host_dip_xr, std::vector<double>& host_dip_xi,
    std::vector<double>& host_dip_yr, std::vector<double>& host_dip_yi,
    std::vector<double>& host_dip_zr, std::vector<double>& host_dip_zi) const
{
    if (num_particles == 0 || d_dipoles == nullptr) {
        throw std::runtime_error("Direct_Electric_Field: dipoles have not been allocated on GPU yet.");
    }

    host_dip_xr.resize(num_particles);
    host_dip_xi.resize(num_particles);
    host_dip_yr.resize(num_particles);
    host_dip_yi.resize(num_particles);
    host_dip_zr.resize(num_particles);
    host_dip_zi.resize(num_particles);

    std::vector<double> host_dips(num_particles * 3 * 2);
    size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(host_dips.data(), d_dipoles, size_dipoles_bytes, cudaMemcpyDeviceToHost));

    for (size_t p = 0; p < num_particles; ++p) {
        host_dip_xr[p] = host_dips[(p * 3 + 0) * 2 + 0];
        host_dip_xi[p] = host_dips[(p * 3 + 0) * 2 + 1];
        host_dip_yr[p] = host_dips[(p * 3 + 1) * 2 + 0];
        host_dip_yi[p] = host_dips[(p * 3 + 1) * 2 + 1];
        host_dip_zr[p] = host_dips[(p * 3 + 2) * 2 + 0];
        host_dip_zi[p] = host_dips[(p * 3 + 2) * 2 + 1];
    }
}

void Direct_Electric_Field::setSelfCoef(
    const std::vector<double>& self_coef_r,
    const std::vector<double>& self_coef_i)
{
    if (self_coef_r.size() != num_particles || self_coef_i.size() != num_particles) {
        throw std::invalid_argument("Direct_Electric_Field: self_coef size must equal num_particles.");
    }
    CUDA_CHECK(cudaMemcpy(d_self_coef_r, self_coef_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, self_coef_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
}

void Direct_Electric_Field::electricField() {
    if (num_particles == 0) return;

    int threadsPerBlock = DIRECT_TILE_SIZE;

    if (num_field_points > 0) {
        size_t E_point_size_complex = num_field_points * 3 + num_quads * 5;
        CUDA_CHECK(cudaMemset(d_E_point, 0, E_point_size_complex * 2 * sizeof(double)));

        int blocksPerGrid   = (static_cast<int>(num_field_points) + threadsPerBlock - 1) / threadsPerBlock;
        direct_field_kernel<<<blocksPerGrid, threadsPerBlock>>>(
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

        int blocksPerGrid   = (static_cast<int>(num_particles) + threadsPerBlock - 1) / threadsPerBlock;
        direct_field_kernel<<<blocksPerGrid, threadsPerBlock>>>(
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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Direct_Electric_Field::calculate() {
    particles_updated = false;
    field_points_updated = false;
    electricField();
}

std::vector<Complex> Direct_Electric_Field::getEPointHost() const {
    size_t num_fields = (num_field_points > 0) ? num_field_points : num_particles;
    size_t size = num_fields * 3 + num_quads * 5;
    if (size == 0 || d_E_point == nullptr) return {};
    std::vector<double> temp(size * 2);
    CUDA_CHECK(cudaMemcpy(temp.data(), d_E_point, size * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<Complex> res(size);
    for (size_t i = 0; i < size; ++i) {
        res[i] = Complex(temp[i * 2], temp[i * 2 + 1]);
    }
    return res;
}
