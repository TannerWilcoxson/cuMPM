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

// Tile size for shared-memory staging of source particles
#define DIRECT_TILE_SIZE 128

// ---------------------------------------------------------------------------
// direct_field_kernel
//
// Each thread handles one TARGET particle j and accumulates contributions from
// all SOURCE particles i != j using the quasi-static free-space dipole tensor
// in the same convention as the Ewald solver:
//
//   E_j += (3*(p_i . r_hat)*r_hat - p_i) / (4*pi*r^3),   r_vec = r_j - r_i
//
// The 1/(4*pi) factor matches the Ewald real-space table normalization
// (fd1_reg = -1/(4*pi*r^3) for the perpendicular component at large r).
//
// Then adds the self-interaction:
//   E_j += (sc_r + i*sc_i) * p_j
//
// Data layout: d_dipoles / d_E_point are interleaved complex doubles indexed
// as double2 at [particle*3 + component], matching the Ewald solver convention.
// ---------------------------------------------------------------------------
__global__ void direct_field_kernel(
    const double* __restrict__ d_dipoles,
    const double* __restrict__ d_x_part,
    const double* __restrict__ d_y_part,
    const double* __restrict__ d_z_part,
    const double* __restrict__ d_self_coef_r,
    const double* __restrict__ d_self_coef_i,
    const double* __restrict__ d_radius,
    double*       d_E_point,
    int           N)
{
    __shared__ double sh_x[DIRECT_TILE_SIZE];
    __shared__ double sh_y[DIRECT_TILE_SIZE];
    __shared__ double sh_z[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_xr[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_xi[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_yr[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_yi[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_zr[DIRECT_TILE_SIZE];
    __shared__ double sh_dip_zi[DIRECT_TILE_SIZE];

    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double E_xr = 0.0, E_xi = 0.0;
    double E_yr = 0.0, E_yi = 0.0;
    double E_zr = 0.0, E_zi = 0.0;

    double jx = (j < N) ? d_x_part[j] : 0.0;
    double jy = (j < N) ? d_y_part[j] : 0.0;
    double jz = (j < N) ? d_z_part[j] : 0.0;

    const double2* dips = reinterpret_cast<const double2*>(d_dipoles);

    // Normalization: T_ij = (3*(p.r_hat)*r_hat - p) / (4*pi*r^3)
    // matches the Ewald real-space table convention where
    // fd1 -> -1/(4*pi*r^3) for the perp component at large r
    const double INV_4PI = 1.0 / (4.0 * 3.14159265358979323846);

    for (int tile_start = 0; tile_start < N; tile_start += DIRECT_TILE_SIZE) {
        int src = tile_start + tid;

        if (src < N) {
            sh_x[tid] = d_x_part[src];
            sh_y[tid] = d_y_part[src];
            sh_z[tid] = d_z_part[src];
            double2 px = dips[src * 3 + 0];
            double2 py = dips[src * 3 + 1];
            double2 pz = dips[src * 3 + 2];
            sh_dip_xr[tid] = px.x;  sh_dip_xi[tid] = px.y;
            sh_dip_yr[tid] = py.x;  sh_dip_yi[tid] = py.y;
            sh_dip_zr[tid] = pz.x;  sh_dip_zi[tid] = pz.y;
        } else {
            sh_x[tid] = 0.0; sh_y[tid] = 0.0; sh_z[tid] = 0.0;
            sh_dip_xr[tid] = 0.0; sh_dip_xi[tid] = 0.0;
            sh_dip_yr[tid] = 0.0; sh_dip_yi[tid] = 0.0;
            sh_dip_zr[tid] = 0.0; sh_dip_zi[tid] = 0.0;
        }
        __syncthreads();

        if (j < N) {
            int tile_end = min(DIRECT_TILE_SIZE, N - tile_start);
            for (int k = 0; k < tile_end; ++k) {
                int i = tile_start + k;
                if (i == j) continue;

                double rx = jx - sh_x[k];
                double ry = jy - sh_y[k];
                double rz = jz - sh_z[k];
                double r2 = rx*rx + ry*ry + rz*rz;
                double r  = sqrt(r2);
                double r3 = r2 * r;
                double inv_r3 = INV_4PI / r3;

                // p_i . r_vec  (real and imaginary parts separately)
                double p_dot_r_r = sh_dip_xr[k]*rx + sh_dip_yr[k]*ry + sh_dip_zr[k]*rz;
                double p_dot_r_i = sh_dip_xi[k]*rx + sh_dip_yi[k]*ry + sh_dip_zi[k]*rz;

                // 3*(p.r_hat)*r_hat / r^3  <=>  3*(p.r_vec)/r^5 * r_vec
                double f3r = 3.0 * p_dot_r_r / r2;
                double f3i = 3.0 * p_dot_r_i / r2;

                E_xr -= inv_r3 * (f3r * rx - sh_dip_xr[k]);
                E_xi -= inv_r3 * (f3i * rx - sh_dip_xi[k]);
                E_yr -= inv_r3 * (f3r * ry - sh_dip_yr[k]);
                E_yi -= inv_r3 * (f3i * ry - sh_dip_yi[k]);
                E_zr -= inv_r3 * (f3r * rz - sh_dip_zr[k]);
                E_zi -= inv_r3 * (f3i * rz - sh_dip_zi[k]);
            }
        }
        __syncthreads();
    }

    if (j < N) {
        double r_j = d_radius[j];
        double self_corr = INV_4PI / (r_j * r_j * r_j);
        double sc_r = d_self_coef_r[j] + self_corr;
        double sc_i = d_self_coef_i[j];
        double2 pj_x = dips[j * 3 + 0];
        double2 pj_y = dips[j * 3 + 1];
        double2 pj_z = dips[j * 3 + 2];

        // complex multiply: (sc_r + i*sc_i) * (p_r + i*p_i)
        E_xr += sc_r * pj_x.x - sc_i * pj_x.y;
        E_xi += sc_r * pj_x.y + sc_i * pj_x.x;
        E_yr += sc_r * pj_y.x - sc_i * pj_y.y;
        E_yi += sc_r * pj_y.y + sc_i * pj_y.x;
        E_zr += sc_r * pj_z.x - sc_i * pj_z.y;
        E_zi += sc_r * pj_z.y + sc_i * pj_z.x;

        double2* out   = reinterpret_cast<double2*>(d_E_point);
        out[j * 3 + 0] = {E_xr, E_xi};
        out[j * 3 + 1] = {E_yr, E_yi};
        out[j * 3 + 2] = {E_zr, E_zi};
    }
}

// ---------------------------------------------------------------------------
// direct_field_evaluate_kernel
//
// Computes all-pairs dipole field evaluation at arbitrary target coordinates
// d_x_field, d_y_field, d_z_field (M points) due to source dipoles (N source particles).
// No self-interaction coefficients or self-interaction exclusions are applied,
// except we skip overlaps if they are exactly at the same point.
// ---------------------------------------------------------------------------
__global__ void direct_field_evaluate_kernel(
    const double* __restrict__ d_dipoles,
    const double* __restrict__ d_x_part,
    const double* __restrict__ d_y_part,
    const double* __restrict__ d_z_part,
    const double* __restrict__ d_x_field,
    const double* __restrict__ d_y_field,
    const double* __restrict__ d_z_field,
    const double* __restrict__ d_radius,
    double*       d_E_point,
    int           N,
    int           M)
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

    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double E_xr = 0.0, E_xi = 0.0;
    double E_yr = 0.0, E_yi = 0.0;
    double E_zr = 0.0, E_zi = 0.0;

    double jx = (j < M) ? d_x_field[j] : 0.0;
    double jy = (j < M) ? d_y_field[j] : 0.0;
    double jz = (j < M) ? d_z_field[j] : 0.0;

    const double2* dips = reinterpret_cast<const double2*>(d_dipoles);
    const double INV_4PI = 1.0 / (4.0 * 3.14159265358979323846);

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
        } else {
            sh_x[tid] = 0.0; sh_y[tid] = 0.0; sh_z[tid] = 0.0;
            sh_radius[tid] = 0.0;
            sh_dip_xr[tid] = 0.0; sh_dip_xi[tid] = 0.0;
            sh_dip_yr[tid] = 0.0; sh_dip_yi[tid] = 0.0;
            sh_dip_zr[tid] = 0.0; sh_dip_zi[tid] = 0.0;
        }
        __syncthreads();

        if (j < M) {
            int tile_end = min(DIRECT_TILE_SIZE, N - tile_start);
            for (int k = 0; k < tile_end; ++k) {
                double rx = jx - sh_x[k];
                double ry = jy - sh_y[k];
                double rz = jz - sh_z[k];
                double r2 = rx*rx + ry*ry + rz*rz;

                // Soften/regularize field evaluation inside particle volume
                double r_sub = sh_radius[k];
                double min_dist = 1.0 * r_sub;
                double min_dist2 = min_dist * min_dist;
                if (r2 < min_dist2) {
                    r2 = min_dist2;
                }

                if (r2 < 1e-18) continue; // Skip exact overlaps to avoid singularity

                double r  = sqrt(r2);
                double r3 = r2 * r;
                double inv_r3 = INV_4PI / r3;

                double p_dot_r_r = sh_dip_xr[k]*rx + sh_dip_yr[k]*ry + sh_dip_zr[k]*rz;
                double p_dot_r_i = sh_dip_xi[k]*rx + sh_dip_yi[k]*ry + sh_dip_zi[k]*rz;

                double f3r = 3.0 * p_dot_r_r / r2;
                double f3i = 3.0 * p_dot_r_i / r2;

                E_xr -= inv_r3 * (f3r * rx - sh_dip_xr[k]);
                E_xi -= inv_r3 * (f3i * rx - sh_dip_xi[k]);
                E_yr -= inv_r3 * (f3r * ry - sh_dip_yr[k]);
                E_yi -= inv_r3 * (f3i * ry - sh_dip_yi[k]);
                E_zr -= inv_r3 * (f3r * rz - sh_dip_zr[k]);
                E_zi -= inv_r3 * (f3i * rz - sh_dip_zi[k]);
            }
        }
        __syncthreads();
    }

    if (j < M) {
        double2* out   = reinterpret_cast<double2*>(d_E_point);
        out[j * 3 + 0] = {E_xr, E_xi};
        out[j * 3 + 1] = {E_yr, E_yi};
        out[j * 3 + 2] = {E_zr, E_zi};
    }
}

// ---------------------------------------------------------------------------
// Direct_Electric_Field host implementation
// ---------------------------------------------------------------------------

Direct_Electric_Field::Direct_Electric_Field(const std::vector<double>& radius)
    : host_radius(radius) {}

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

    if (N != num_particles) {
        if (d_x_part)      { cudaFree(d_x_part);      d_x_part      = nullptr; }
        if (d_y_part)      { cudaFree(d_y_part);      d_y_part      = nullptr; }
        if (d_z_part)      { cudaFree(d_z_part);      d_z_part      = nullptr; }
        if (d_dipoles)     { cudaFree(d_dipoles);     d_dipoles     = nullptr; }
        if (d_self_coef_r) { cudaFree(d_self_coef_r); d_self_coef_r = nullptr; }
        if (d_self_coef_i) { cudaFree(d_self_coef_i); d_self_coef_i = nullptr; }
        if (d_radius)      { cudaFree(d_radius);      d_radius      = nullptr; }
        if (num_field_points == 0 && d_E_point) {
            cudaFree(d_E_point);
            d_E_point = nullptr;
        }

        num_particles = N;
        CUDA_CHECK(cudaMalloc(&d_x_part,      N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y_part,      N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z_part,      N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dipoles,     N * 3 * 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_r, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_radius,      N * sizeof(double)));

        CUDA_CHECK(cudaMemset(d_dipoles,     0, N * 3 * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_self_coef_r, 0, N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_self_coef_i, 0, N * sizeof(double)));

        if (num_field_points == 0) {
            CUDA_CHECK(cudaMalloc(&d_E_point, N * 3 * 2 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_E_point, 0, N * 3 * 2 * sizeof(double)));
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
        
        // If we clear field coordinates, we must restore d_E_point to fit the particles if needed
        if (num_particles > 0) {
            CUDA_CHECK(cudaMalloc(&d_E_point, num_particles * 3 * 2 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_E_point, 0, num_particles * 3 * 2 * sizeof(double)));
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
        CUDA_CHECK(cudaMalloc(&d_E_point, M * 3 * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_E_point, 0, M * 3 * 2 * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(d_x_field, x_field.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_field, y_field.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_field, z_field.data(), M * sizeof(double), cudaMemcpyHostToDevice));

    field_points_updated = true;
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
        CUDA_CHECK(cudaMemset(d_E_point, 0, num_field_points * 3 * 2 * sizeof(double)));

        int blocksPerGrid   = (static_cast<int>(num_field_points) + threadsPerBlock - 1) / threadsPerBlock;
        direct_field_evaluate_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_dipoles,
            d_x_part, d_y_part, d_z_part,
            d_x_field, d_y_field, d_z_field,
            d_radius,
            d_E_point,
            static_cast<int>(num_particles),
            static_cast<int>(num_field_points)
        );
    } else {
        CUDA_CHECK(cudaMemset(d_E_point, 0, num_particles * 3 * 2 * sizeof(double)));

        int blocksPerGrid   = (static_cast<int>(num_particles) + threadsPerBlock - 1) / threadsPerBlock;
        direct_field_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_dipoles,
            d_x_part, d_y_part, d_z_part,
            d_self_coef_r, d_self_coef_i,
            d_radius,
            d_E_point,
            static_cast<int>(num_particles)
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
