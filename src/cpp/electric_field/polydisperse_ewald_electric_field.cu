#include "polydisperse_ewald_electric_field.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <chrono>

// CUDA Error Checking macro
#ifndef CUDA_CHECK
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
      if (abort) exit(code);
   }
}
#endif

// -----------------------------------------------------------------------------
// CUDA Kernels for Polydisperse Ewald Solver - incremental build comment updated
// -----------------------------------------------------------------------------


Polydisperse_Ewald_Electric_Field::Polydisperse_Ewald_Electric_Field(
    double box_x, double box_y, double box_z,
    double errortol,
    double xi,
    FieldCalcMode mode,
    const std::vector<double>& particle_radii,
    bool solve_quadrupoles,
    const std::vector<int>& quad_idxs)
    : Ewald_Electric_Field_Base(box_x, box_y, box_z, errortol, xi, mode, solve_quadrupoles, quad_idxs),
      h_radii(particle_radii)
{
    num_particles = h_radii.size();

    // 1. Identify unique radii
    for (double r : h_radii) {
        bool exists = false;
        for (double ur : unique_radii) {
            if (std::abs(ur - r) < 1e-5) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            unique_radii.push_back(r);
        }
    }
    std::sort(unique_radii.begin(), unique_radii.end());
    num_unique_radii = unique_radii.size();
    num_pairs_unique = num_unique_radii * (num_unique_radii + 1) / 2;

    // 2. Build diagonal mappings
    std::vector<int> h_col_ind(num_unique_radii * num_unique_radii, 0);
    int lin = 0;
    for (size_t i = 0; i < num_unique_radii; ++i) {
        for (size_t j = i; j < num_unique_radii; ++j) {
            h_col_ind[i * num_unique_radii + j] = lin;
            h_col_ind[j * num_unique_radii + i] = lin;
            lin++;
        }
    }

    std::vector<int> h_radius_idx(num_particles, 0);
    for (size_t i = 0; i < num_particles; ++i) {
        double r = h_radii[i];
        for (size_t k = 0; k < num_unique_radii; ++k) {
            if (std::abs(unique_radii[k] - r) < 1e-5) {
                h_radius_idx[i] = k;
                break;
            }
        }
    }

    // 3. Allocate and copy metadata to GPU
    CUDA_CHECK(cudaMalloc(&d_radii, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_radius_idx, num_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, num_unique_radii * num_unique_radii * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_radii, h_radii.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radius_idx, h_radius_idx.data(), num_particles * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, h_col_ind.data(), num_unique_radii * num_unique_radii * sizeof(int), cudaMemcpyHostToDevice));

    // 4. Compute Ewald parameters and tables
    computePrecalculations();
    computeRealSpaceTables();

    // 5. Allocate scalar grids and FFT plan
    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    CUDA_CHECK(cudaMalloc(&d_fE_grid, grid_voxels * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fEs_grid, grid_voxels * 2 * sizeof(double)));

    cufftResult plan_res = cufftPlan3d((cufftHandle*)&fft_plan, num_grid[0], num_grid[1], num_grid[2], CUFFT_Z2Z);
    if (plan_res != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT 3D plan creation failed with code: " + std::to_string(plan_res));
    }

    if (solve_quadrupoles) {
        CUDA_CHECK(cudaMalloc(&d_fG_grid, grid_voxels * 5 * 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_fGs_grid, grid_voxels * 5 * 2 * sizeof(double)));

        int n[3] = { num_grid[0], num_grid[1], num_grid[2] };
        cufftResult plan_res_G = cufftPlanMany((cufftHandle*)&fft_plan_G, 3, n,
                                              n, 5, 1, // inembed, istride, idist
                                              n, 5, 1, // onembed, ostride, odist
                                              CUFFT_Z2Z, 5);
        if (plan_res_G != CUFFT_SUCCESS) {
            throw std::runtime_error("cuFFT plan G creation failed with code: " + std::to_string(plan_res_G));
        }
    }
}

Polydisperse_Ewald_Electric_Field::~Polydisperse_Ewald_Electric_Field() {
    if (d_radii) cudaFree(d_radii);
    if (d_radius_idx) cudaFree(d_radius_idx);
    if (d_col_ind) cudaFree(d_col_ind);
    if (d_self_perp_uniq) cudaFree(d_self_perp_uniq);
    if (d_self_G2_uniq) cudaFree(d_self_G2_uniq);
    if (d_spread_coef_Q) cudaFree(d_spread_coef_Q);
    if (d_contract_coef_Q) cudaFree(d_contract_coef_Q);
}

void Polydisperse_Ewald_Electric_Field::updateParticleCoordinates(
    const std::vector<double>& x_part,
    const std::vector<double>& y_part,
    const std::vector<double>& z_part)
{
    Ewald_Electric_Field_Base::updateParticleCoordinates(x_part, y_part, z_part);
}

void Polydisperse_Ewald_Electric_Field::getPrecalculationsHost(std::vector<int>& host_offset,
                                            std::vector<double>& host_offsetxyz,
                                            std::vector<double>& host_scale_coef) const {
    if (num_offsets == 0 || d_offset == nullptr || d_offsetxyz == nullptr || 
        d_scale_coef == nullptr) {
        throw std::runtime_error("Ewald precalculations have not been calculated/allocated on GPU yet.");
    }

    host_offset.resize(num_offsets * 3);
    host_offsetxyz.resize(num_offsets * 3);
    
    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    host_scale_coef.resize(grid_voxels);

    CUDA_CHECK(cudaMemcpy(host_offset.data(), d_offset, num_offsets * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_offsetxyz.data(), d_offsetxyz, num_offsets * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_scale_coef.data(), d_scale_coef, grid_voxels * sizeof(double), cudaMemcpyDeviceToHost));
}
__device__ static double interpolate_table_gpu_polydisperse(
    double r,
    const double* r_table,
    const double* y_table,
    size_t table_size,
    int col,
    int num_cols)
{
    if (r <= 0.0) return y_table[0 * num_cols + col];
    if (r >= r_table[table_size - 1]) return y_table[(table_size - 1) * num_cols + col];

    if (r < 1.0) {
        double t = r;
        return y_table[0 * num_cols + col] * (1.0 - t) + y_table[1 * num_cols + col] * t;
    }

    size_t idx = 1 + static_cast<size_t>((r - 1.0) / 0.001);
    if (idx >= table_size - 1) {
        return y_table[(table_size - 1) * num_cols + col];
    }
    double r0 = r_table[idx];
    double r1 = r_table[idx + 1];
    double t = (r - r0) / (r1 - r0);
    return y_table[idx * num_cols + col] * (1.0 - t) + y_table[(idx + 1) * num_cols + col] * t;
}



__global__ void spread_precalcs_kernel_polydisperse(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
    const double* __restrict__ radii,
    const int* __restrict__ offset,
    const double* __restrict__ offsetxyz,
    double* __restrict__ spread_coef,
    int* __restrict__ spread_idxs,
    size_t num_particles,
    size_t num_offsets,
    double grid_spacing_x,
    double grid_spacing_y,
    double grid_spacing_z,
    int num_grid_x,
    int num_grid_y,
    int num_grid_z,
    double eta,
    double xi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles * num_offsets) return;

    int i = idx / num_offsets;
    int o = idx % num_offsets;

    double px = x_part[i];
    double py = y_part[i];
    double pz = z_part[i];
    double a_i = radii[i];

    int gix = static_cast<int>(round(px / grid_spacing_x));
    int giy = static_cast<int>(round(py / grid_spacing_y));
    int giz = static_cast<int>(round(pz / grid_spacing_z));

    double dx = gix * grid_spacing_x - px;
    double dy = giy * grid_spacing_y - py;
    double dz = giz * grid_spacing_z - pz;

    int ox = offset[o * 3 + 0];
    int oy = offset[o * 3 + 1];
    int oz = offset[o * 3 + 2];

    double oxyz_x = offsetxyz[o * 3 + 0];
    double oxyz_y = offsetxyz[o * 3 + 1];
    double oxyz_z = offsetxyz[o * 3 + 2];

    spread_idxs[idx] = (((gix + ox + 256) & 0x3FF) << 20) | (((giy + oy + 256) & 0x3FF) << 10) | ((giz + oz + 256) & 0x3FF);

    double gdx = dx + oxyz_x;
    double gdy = dy + oxyz_y;
    double gdz = dz + oxyz_z;

    double d = sqrt(gdx * gdx + gdy * gdy + gdz * gdz);

    const double PI = 3.14159265358979323846;
    double xi_sq = xi * xi;

    if (d >= 1e-6) {
        double k = 3.0 / (8.0 * sqrt(2.0) * pow(PI, 1.5) * pow(a_i, 3.0) * sqrt(eta) * xi * d * d);
        double term_p = (eta + 4.0 * a_i * xi_sq * d) * exp(-2.0 * pow(d + a_i, 2) * xi_sq / eta);
        double term_m = (eta - 4.0 * a_i * xi_sq * d) * exp(-2.0 * pow(d - a_i, 2) * xi_sq / eta);
        double coef_scalar = k * (term_p - term_m);

        spread_coef[idx * 3 + 0] = coef_scalar * (gdx / d);
        spread_coef[idx * 3 + 1] = coef_scalar * (gdy / d);
        spread_coef[idx * 3 + 2] = coef_scalar * (gdz / d);
    } else {
        double k_async = 8.0 * sqrt(2.0) * pow(xi, 5.0) * exp(-2.0 * a_i * xi_sq / eta) / (pow(PI, 1.5) * pow(eta, 2.5));
        spread_coef[idx * 3 + 0] = k_async * gdx;
        spread_coef[idx * 3 + 1] = k_async * gdy;
        spread_coef[idx * 3 + 2] = k_async * gdz;
    }
}

__global__ void contract_precalcs_kernel_polydisperse(
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
    const double* __restrict__ radii,
    const int* __restrict__ offset,
    const double* __restrict__ offsetxyz,
    double* __restrict__ contract_coef,
    int* __restrict__ contract_idxs,
    int* __restrict__ particle_index,
    size_t num_field_points,
    size_t num_particles,
    size_t num_offsets,
    double grid_spacing_x,
    double grid_spacing_y,
    double grid_spacing_z,
    int num_grid_x,
    int num_grid_y,
    int num_grid_z,
    double eta,
    double xi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_field_points * num_offsets) return;

    int i = idx / num_offsets;
    int o = idx % num_offsets;

    double px = x_field[i];
    double py = y_field[i];
    double pz = z_field[i];
    double a_i = (i < num_particles) ? radii[i] : radii[0];

    int gix = static_cast<int>(round(px / grid_spacing_x));
    int giy = static_cast<int>(round(py / grid_spacing_y));
    int giz = static_cast<int>(round(pz / grid_spacing_z));

    double dx = gix * grid_spacing_x - px;
    double dy = giy * grid_spacing_y - py;
    double dz = giz * grid_spacing_z - pz;

    int ox = offset[o * 3 + 0];
    int oy = offset[o * 3 + 1];
    int oz = offset[o * 3 + 2];

    double oxyz_x = offsetxyz[o * 3 + 0];
    double oxyz_y = offsetxyz[o * 3 + 1];
    double oxyz_z = offsetxyz[o * 3 + 2];

    int geix = ((gix + ox) % num_grid_x + num_grid_x) % num_grid_x;
    int geiy = ((giy + oy) % num_grid_y + num_grid_y) % num_grid_y;
    int geiz = ((giz + oz) % num_grid_z + num_grid_z) % num_grid_z;

    int transposed_idx = o * num_field_points + i;
    contract_idxs[transposed_idx] = geix * num_grid_y * num_grid_z + geiy * num_grid_z + geiz;

    particle_index[transposed_idx] = i;

    double gdx = dx + oxyz_x;
    double gdy = dy + oxyz_y;
    double gdz = dz + oxyz_z;

    double d = sqrt(gdx * gdx + gdy * gdy + gdz * gdz);

    const double PI = 3.14159265358979323846;
    double xi_sq = xi * xi;

    size_t num_contract = num_field_points * num_offsets;

    if (d >= 1e-6) {
        double k = 3.0 / (8.0 * sqrt(2.0) * pow(PI, 1.5) * pow(a_i, 3.0) * sqrt(eta) * xi * d * d);
        double term_p = (eta + 4.0 * a_i * xi_sq * d) * exp(-2.0 * pow(d + a_i, 2) * xi_sq / eta);
        double term_m = (eta - 4.0 * a_i * xi_sq * d) * exp(-2.0 * pow(d - a_i, 2) * xi_sq / eta);
        double coef_scalar = k * (term_p - term_m);

        contract_coef[0 * num_contract + transposed_idx] = coef_scalar * (gdx / d);
        contract_coef[1 * num_contract + transposed_idx] = coef_scalar * (gdy / d);
        contract_coef[2 * num_contract + transposed_idx] = coef_scalar * (gdz / d);
    } else {
        double k_async = 8.0 * sqrt(2.0) * pow(xi, 5.0) * exp(-2.0 * a_i * xi_sq / eta) / (pow(PI, 1.5) * pow(eta, 2.5));
        contract_coef[0 * num_contract + transposed_idx] = k_async * gdx;
        contract_coef[1 * num_contract + transposed_idx] = k_async * gdy;
        contract_coef[2 * num_contract + transposed_idx] = k_async * gdz;
    }
}

__global__ void real_space_precalcs_kernel_polydisperse(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
    const int* __restrict__ radius_idx,
    const int* __restrict__ col_ind,
    int num_cols_unique,
    const int* __restrict__ neighbor_list,
    const int* __restrict__ neighbor_counts,
    const int* __restrict__ particle_offsets,
    const double* __restrict__ r_table,
    const double* __restrict__ field_dip_1,
    const double* __restrict__ field_dip_2,
    size_t table_size,
    double* __restrict__ perp,
    double* __restrict__ para,
    size_t num_particles,
    int max_neighbors,
    double box_x,
    double box_y,
    double box_z,
    double rc,
    bool solve_quadrupoles,
    const double* __restrict__ field_quad_1,
    const double* __restrict__ field_quad_2,
    const double* __restrict__ field_quad_3,
    const double* __restrict__ grad_quad_1,
    const double* __restrict__ grad_quad_2,
    const double* __restrict__ grad_quad_3,
    const double* __restrict__ grad_quad_4,
    double* __restrict__ perp_Q,
    double* __restrict__ para_Q,
    double* __restrict__ Q3,
    double* __restrict__ G1,
    double* __restrict__ G2,
    double* __restrict__ G3,
    double* __restrict__ G4)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    int count = neighbor_counts[i];
    if (count > max_neighbors) count = max_neighbors;
    int start_idx = particle_offsets[i];

    double xi = x_part[i];
    double yi = y_part[i];
    double zi = z_part[i];
    int radius_idx_i = radius_idx[i];

    for (int k = 0; k < count; ++k) {
        int j = neighbor_list[i * max_neighbors + k];

        double rx = x_field[j] - xi;
        double ry = y_field[j] - yi;
        double rz = z_field[j] - zi;

        if (box_x > 0.0) rx -= box_x * round(rx / box_x);
        if (box_y > 0.0) ry -= box_y * round(ry / box_y);
        if (box_z > 0.0) rz -= box_z * round(rz / box_z);

        double d = sqrt(rx * rx + ry * ry + rz * rz);

        if (d < rc) {
            double d_eff = d < 1.0 ? 1.0 : d;
            int radius_idx_j = (j < num_particles) ? radius_idx[j] : 0;
            int col = col_ind[radius_idx_i * num_cols_unique + radius_idx_j];
            int num_cols_total = num_cols_unique * (num_cols_unique + 1) / 2;

            double p_val = interpolate_table_gpu_polydisperse(d_eff, r_table, field_dip_1, table_size, col, num_cols_total);
            double a_val = interpolate_table_gpu_polydisperse(d_eff, r_table, field_dip_2, table_size, col, num_cols_total);

            perp[start_idx + k] = p_val;
            para[start_idx + k] = a_val;

            if (solve_quadrupoles) {
                perp_Q[start_idx + k] = interpolate_table_gpu_polydisperse(d_eff, r_table, field_quad_1, table_size, 0, 1);
                para_Q[start_idx + k] = interpolate_table_gpu_polydisperse(d_eff, r_table, field_quad_2, table_size, 0, 1);
                Q3[start_idx + k]     = interpolate_table_gpu_polydisperse(d_eff, r_table, field_quad_3, table_size, 0, 1);
                G1[start_idx + k]     = interpolate_table_gpu_polydisperse(d_eff, r_table, grad_quad_1, table_size, 0, 1);
                G2[start_idx + k]     = interpolate_table_gpu_polydisperse(d_eff, r_table, grad_quad_2, table_size, 0, 1);
                G3[start_idx + k]     = interpolate_table_gpu_polydisperse(d_eff, r_table, grad_quad_3, table_size, 0, 1);
                G4[start_idx + k]     = interpolate_table_gpu_polydisperse(d_eff, r_table, grad_quad_4, table_size, 0, 1);
            }
        }
    }
}

__global__ void spread_kernel_polydisperse(
    const double* __restrict__ d_dipoles,
    const double* __restrict__ spread_coef,
    const int* __restrict__ spread_idxs,
    double* __restrict__ fE_grid,
    size_t num_spread,
    size_t num_offsets,
    int num_grid_x,
    int num_grid_y,
    int num_grid_z)
{
    __shared__ int block_min_gx, block_max_gx;
    __shared__ int block_min_gy, block_max_gy;
    __shared__ int block_min_gz, block_max_gz;

    if (threadIdx.x == 0) {
        block_min_gx = 999999; block_max_gx = -999999;
        block_min_gy = 999999; block_max_gy = -999999;
        block_min_gz = 999999; block_max_gz = -999999;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (idx < num_spread);

    int gx = 0, gy = 0, gz = 0;
    double val_r = 0.0, val_i = 0.0;

    if (active) {
        int i = idx / num_offsets;

        const double2* d_dipoles_d2 = reinterpret_cast<const double2*>(d_dipoles);
        double2 dip_x = d_dipoles_d2[i * 3 + 0];
        double2 dip_y = d_dipoles_d2[i * 3 + 1];
        double2 dip_z = d_dipoles_d2[i * 3 + 2];

        double px_r = dip_x.x;
        double px_i = dip_x.y;
        double py_r = dip_y.x;
        double py_i = dip_y.y;
        double pz_r = dip_z.x;
        double pz_i = dip_z.y;

        double cx = spread_coef[idx * 3 + 0];
        double cy = spread_coef[idx * 3 + 1];
        double cz = spread_coef[idx * 3 + 2];

        // Dot product: C . p
        val_r = cx * px_r + cy * py_r + cz * pz_r;
        val_i = cx * px_i + cy * py_i + cz * pz_i;

        uint32_t packed = spread_idxs[idx];
        gx = static_cast<int>(packed >> 20) - 256;
        gy = static_cast<int>((packed >> 10) & 0x3FF) - 256;
        gz = static_cast<int>(packed & 0x3FF) - 256;

        atomicMin(&block_min_gx, gx);
        atomicMax(&block_max_gx, gx);
        atomicMin(&block_min_gy, gy);
        atomicMax(&block_max_gy, gy);
        atomicMin(&block_min_gz, gz);
        atomicMax(&block_max_gz, gz);
    }
    __syncthreads();

    int dim_x = block_max_gx - block_min_gx + 1;
    int dim_y = block_max_gy - block_min_gy + 1;
    int dim_z = block_max_gz - block_min_gz + 1;
    int local_grid_size = dim_x * dim_y * dim_z;

    extern __shared__ double s_grid[];

    bool use_shared = (local_grid_size > 0 && local_grid_size <= 512);

    if (use_shared) {
        // Initialize shared memory
        for (int offset = threadIdx.x; offset < local_grid_size * 2; offset += blockDim.x) {
            s_grid[offset] = 0.0;
        }
        __syncthreads();

        if (active) {
            int local_x = gx - block_min_gx;
            int local_y = gy - block_min_gy;
            int local_z = gz - block_min_gz;
            int local_idx = local_x * dim_y * dim_z + local_y * dim_z + local_z;

            atomicAdd(&s_grid[local_idx * 2 + 0], val_r);
            atomicAdd(&s_grid[local_idx * 2 + 1], val_i);
        }
        __syncthreads();

        // Flush to global memory
        for (int offset = threadIdx.x; offset < local_grid_size; offset += blockDim.x) {
            int local_x = offset / (dim_y * dim_z);
            int local_y = (offset / dim_z) % dim_y;
            int local_z = offset % dim_z;

            int global_gx = ((block_min_gx + local_x) % num_grid_x + num_grid_x) % num_grid_x;
            int global_gy = ((block_min_gy + local_y) % num_grid_y + num_grid_y) % num_grid_y;
            int global_gz = ((block_min_gz + local_z) % num_grid_z + num_grid_z) % num_grid_z;

            size_t global_idx = static_cast<size_t>(global_gx) * num_grid_y * num_grid_z +
                                static_cast<size_t>(global_gy) * num_grid_z +
                                static_cast<size_t>(global_gz);

            double v_r = s_grid[offset * 2 + 0];
            double v_i = s_grid[offset * 2 + 1];

            if (v_r != 0.0) atomicAdd(&fE_grid[global_idx * 2 + 0], v_r);
            if (v_i != 0.0) atomicAdd(&fE_grid[global_idx * 2 + 1], v_i);
        }
    } else {
        // Fallback to direct global memory writes
        if (active) {
            int global_gx = ((gx) % num_grid_x + num_grid_x) % num_grid_x;
            int global_gy = ((gy) % num_grid_y + num_grid_y) % num_grid_y;
            int global_gz = ((gz) % num_grid_z + num_grid_z) % num_grid_z;

            size_t global_idx = static_cast<size_t>(global_gx) * num_grid_y * num_grid_z +
                                static_cast<size_t>(global_gy) * num_grid_z +
                                static_cast<size_t>(global_gz);

            atomicAdd(&fE_grid[global_idx * 2 + 0], val_r);
            atomicAdd(&fE_grid[global_idx * 2 + 1], val_i);
        }
    }
}

__global__ void scale_kernel_polydisperse(
    double* __restrict__ fE_grid,
    const double* __restrict__ scale_coef,
    size_t num_voxels)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;

    double sc = scale_coef[v];
    fE_grid[v * 2 + 0] *= sc;
    fE_grid[v * 2 + 1] *= sc;
}

__global__ void contract_kernel_polydisperse(
    const double* __restrict__ fE_grid,
    const double* __restrict__ contract_coef,
    const int* __restrict__ contract_idxs,
    double* __restrict__ E_point,
    size_t num_field_points,
    size_t num_offsets,
    int num_grid_x,
    int num_grid_y,
    int num_grid_z,
    double prod_h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_field_points) return;

    double E_x_r = 0.0, E_x_i = 0.0;
    double E_y_r = 0.0, E_y_i = 0.0;
    double E_z_r = 0.0, E_z_i = 0.0;

    size_t num_contract = num_field_points * num_offsets;

    for (size_t o = 0; o < num_offsets; ++o) {
        size_t idx = o * num_field_points + i;
        int v = contract_idxs[idx];

        double grid_r = fE_grid[v * 2 + 0];
        double grid_i = fE_grid[v * 2 + 1];

        double cx = contract_coef[0 * num_contract + idx];
        double cy = contract_coef[1 * num_contract + idx];
        double cz = contract_coef[2 * num_contract + idx];

        E_x_r += cx * grid_r;
        E_x_i += cx * grid_i;

        E_y_r += cy * grid_r;
        E_y_i += cy * grid_i;

        E_z_r += cz * grid_r;
        E_z_i += cz * grid_i;
    }

    double scale = prod_h;
    E_point[(i * 3 + 0) * 2 + 0] = scale * E_x_r;
    E_point[(i * 3 + 0) * 2 + 1] = scale * E_x_i;
    E_point[(i * 3 + 1) * 2 + 0] = scale * E_y_r;
    E_point[(i * 3 + 1) * 2 + 1] = scale * E_y_i;
    E_point[(i * 3 + 2) * 2 + 0] = scale * E_z_r;
    E_point[(i * 3 + 2) * 2 + 1] = scale * E_z_i;
}

__global__ void real_space_self_kernel_polydisperse(
    const double* __restrict__ d_dipoles,
    const double* __restrict__ d_self_coef_r,
    const double* __restrict__ d_self_coef_i,
    const int* __restrict__ radius_idx,
    const double* __restrict__ d_self_perp_uniq,
    double* __restrict__ E_point,
    size_t num_particles,
    FieldCalcMode mode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    double sc_r = d_self_coef_r[i];
    double sc_i = d_self_coef_i[i];

    int radius_idx_i = radius_idx[i];
    double self_perp_val = d_self_perp_uniq[radius_idx_i];

    double factor_r = sc_r + self_perp_val;
    double factor_i = sc_i;

    const double2* d_dipoles_d2 = reinterpret_cast<const double2*>(d_dipoles);
    double2 dip_x = d_dipoles_d2[i * 3 + 0];
    double2 dip_y = d_dipoles_d2[i * 3 + 1];
    double2 dip_z = d_dipoles_d2[i * 3 + 2];

    double dx_r = dip_x.x;
    double dx_i = dip_x.y;
    double dy_r = dip_y.x;
    double dy_i = dip_y.y;
    double dz_r = dip_z.x;
    double dz_i = dip_z.y;

    atomicAdd(&E_point[(i * 3 + 0) * 2 + 0], factor_r * dx_r - factor_i * dx_i);
    atomicAdd(&E_point[(i * 3 + 0) * 2 + 1], factor_r * dx_i + factor_i * dx_r);

    atomicAdd(&E_point[(i * 3 + 1) * 2 + 0], factor_r * dy_r - factor_i * dy_i);
    atomicAdd(&E_point[(i * 3 + 1) * 2 + 1], factor_r * dy_i + factor_i * dy_r);

    atomicAdd(&E_point[(i * 3 + 2) * 2 + 0], factor_r * dz_r - factor_i * dz_i);
    atomicAdd(&E_point[(i * 3 + 2) * 2 + 1], factor_r * dz_i + factor_i * dz_r);
}

__global__ void real_space_neighbor_kernel_polydisperse(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
    const double* __restrict__ d_dipoles,
    const int* __restrict__ neighbor_list,
    const int* __restrict__ neighbor_counts,
    const int* __restrict__ particle_offsets,
    const double* __restrict__ perp,
    const double* __restrict__ para,
    double* __restrict__ E_point,
    size_t num_particles,
    int max_neighbors,
    double box_x,
    double box_y,
    double box_z,
    double rc,
    FieldCalcMode mode)
{
    double sign = (mode == FieldCalcMode::SOLVER_AX) ? 1.0 : -1.0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    int count = neighbor_counts[i];
    if (count > max_neighbors) count = max_neighbors;
    int start_idx = particle_offsets[i];

    double xi = x_part[i];
    double yi = y_part[i];
    double zi = z_part[i];

    for (int k = 0; k < count; ++k) {
        int j = neighbor_list[i * max_neighbors + k];

        double rx = x_field[j] - xi;
        double ry = y_field[j] - yi;
        double rz = z_field[j] - zi;

        if (box_x > 0.0) rx -= box_x * round(rx / box_x);
        if (box_y > 0.0) ry -= box_y * round(ry / box_y);
        if (box_z > 0.0) rz -= box_z * round(rz / box_z);

        double d = sqrt(rx * rx + ry * ry + rz * rz);

        if (d < rc) {
            double d_eff = d < 1.0 ? 1.0 : d;
            double delta_x = rx / d_eff;
            double delta_y = ry / d_eff;
            double delta_z = rz / d_eff;

            // neighbor dipoles
            const double2* d_dipoles_d2 = reinterpret_cast<const double2*>(d_dipoles);
            double2 dip_x = d_dipoles_d2[i * 3 + 0];
            double2 dip_y = d_dipoles_d2[i * 3 + 1];
            double2 dip_z = d_dipoles_d2[i * 3 + 2];

            double dip_x_r = dip_x.x;
            double dip_x_i = dip_x.y;
            double dip_y_r = dip_y.x;
            double dip_y_i = dip_y.y;
            double dip_z_r = dip_z.x;
            double dip_z_i = dip_z.y;

            double delta_dip_r = dip_x_r * delta_x + dip_y_r * delta_y + dip_z_r * delta_z;
            double delta_dip_i = dip_x_i * delta_x + dip_y_i * delta_y + dip_z_i * delta_z;

            double perp_val = perp[start_idx + k];
            double para_val = para[start_idx + k];

            double contrib_x_r = perp_val * (dip_x_r - delta_x * delta_dip_r) + para_val * delta_x * delta_dip_r;
            double contrib_x_i = perp_val * (dip_x_i - delta_x * delta_dip_i) + para_val * delta_x * delta_dip_i;

            double contrib_y_r = perp_val * (dip_y_r - delta_y * delta_dip_r) + para_val * delta_y * delta_dip_r;
            double contrib_y_i = perp_val * (dip_y_i - delta_y * delta_dip_i) + para_val * delta_y * delta_dip_i;

            double contrib_z_r = perp_val * (dip_z_r - delta_z * delta_dip_r) + para_val * delta_z * delta_dip_r;
            double contrib_z_i = perp_val * (dip_z_i - delta_z * delta_dip_i) + para_val * delta_z * delta_dip_i;

            atomicAdd(&E_point[(j * 3 + 0) * 2 + 0], sign * contrib_x_r);
            atomicAdd(&E_point[(j * 3 + 0) * 2 + 1], sign * contrib_x_i);

            atomicAdd(&E_point[(j * 3 + 1) * 2 + 0], sign * contrib_y_r);
            atomicAdd(&E_point[(j * 3 + 1) * 2 + 1], sign * contrib_y_i);

            atomicAdd(&E_point[(j * 3 + 2) * 2 + 0], sign * contrib_z_r);
            atomicAdd(&E_point[(j * 3 + 2) * 2 + 1], sign * contrib_z_i);
        }
    }
}

__global__ void fftshift_3d_kernel_scalar(
    const double* __restrict__ input,
    double* __restrict__ output,
    int N0, int N1, int N2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_voxels = N0 * N1 * N2;
    if (idx >= num_voxels) return;

    int x = idx / (N1 * N2);
    int y = (idx / N2) % N1;
    int z = idx % N2;

    int shifted_x = (x + N0 / 2) % N0;
    int shifted_y = (y + N1 / 2) % N1;
    int shifted_z = (z + N2 / 2) % N2;

    size_t shifted_idx = static_cast<size_t>(shifted_x) * N1 * N2 +
                          static_cast<size_t>(shifted_y) * N2 +
                          static_cast<size_t>(shifted_z);

    output[shifted_idx * 2 + 0] = input[idx * 2 + 0];
    output[shifted_idx * 2 + 1] = input[idx * 2 + 1];
}

__global__ void ifftshift_3d_kernel_scalar(
    const double* __restrict__ input,
    double* __restrict__ output,
    int N0, int N1, int N2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_voxels = N0 * N1 * N2;
    if (idx >= num_voxels) return;

    int x = idx / (N1 * N2);
    int y = (idx / N2) % N1;
    int z = idx % N2;

    int xs = (x + (N0 + 1) / 2) % N0;
    int ys = (y + (N1 + 1) / 2) % N1;
    int zs = (z + (N2 + 1) / 2) % N2;

    size_t shifted_idx = static_cast<size_t>(xs) * N1 * N2 +
                          static_cast<size_t>(ys) * N2 +
                          static_cast<size_t>(zs);

    double scale_factor = 1.0 / (static_cast<double>(N0) * N1 * N2);

    output[shifted_idx * 2 + 0] = input[idx * 2 + 0] * scale_factor;
    output[shifted_idx * 2 + 1] = input[idx * 2 + 1] * scale_factor;
}

// -----------------------------------------------------------------------------
// Quadrupole-related Kernels for Polydisperse solver
// -----------------------------------------------------------------------------

__global__ void spread_precalcs_kernel_point(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
    const int* __restrict__ offset,
    const double* __restrict__ offsetxyz,
    double* __restrict__ spread_coef_Q,
    size_t num_particles,
    size_t num_offsets,
    double grid_spacing_x,
    double grid_spacing_y,
    double grid_spacing_z,
    double eta,
    double xi,
    double const_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles * num_offsets) return;

    int i = idx / num_offsets;
    int o = idx % num_offsets;

    double px = x_part[i];
    double py = y_part[i];
    double pz = z_part[i];

    int gix = static_cast<int>(round(px / grid_spacing_x));
    int giy = static_cast<int>(round(py / grid_spacing_y));
    int giz = static_cast<int>(round(pz / grid_spacing_z));

    double dx = gix * grid_spacing_x - px;
    double dy = giy * grid_spacing_y - py;
    double dz = giz * grid_spacing_z - pz;

    double oxyz_x = offsetxyz[o * 3 + 0];
    double oxyz_y = offsetxyz[o * 3 + 1];
    double oxyz_z = offsetxyz[o * 3 + 2];

    double gdx = dx + oxyz_x;
    double gdy = dy + oxyz_y;
    double gdz = dz + oxyz_z;

    double d_sq = gdx * gdx + gdy * gdy + gdz * gdz;

    spread_coef_Q[idx] = const_factor * exp(-2.0 * xi * xi * d_sq);
}

__global__ void contract_precalcs_kernel_point(
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
    const int* __restrict__ offset,
    const double* __restrict__ offsetxyz,
    double* __restrict__ contract_coef_Q,
    size_t num_field_points,
    size_t num_offsets,
    double grid_spacing_x,
    double grid_spacing_y,
    double grid_spacing_z,
    double eta,
    double xi,
    double const_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_field_points * num_offsets) return;

    int i = idx / num_offsets;
    int o = idx % num_offsets;

    double px = x_field[i];
    double py = y_field[i];
    double pz = z_field[i];

    int gix = static_cast<int>(round(px / grid_spacing_x));
    int giy = static_cast<int>(round(py / grid_spacing_y));
    int giz = static_cast<int>(round(pz / grid_spacing_z));

    double dx = gix * grid_spacing_x - px;
    double dy = giy * grid_spacing_y - py;
    double dz = giz * grid_spacing_z - pz;

    double oxyz_x = offsetxyz[o * 3 + 0];
    double oxyz_y = offsetxyz[o * 3 + 1];
    double oxyz_z = offsetxyz[o * 3 + 2];

    double gdx = dx + oxyz_x;
    double gdy = dy + oxyz_y;
    double gdz = dz + oxyz_z;

    double d_sq = gdx * gdx + gdy * gdy + gdz * gdz;

    int transposed_idx = o * num_field_points + i;
    contract_coef_Q[transposed_idx] = const_factor * exp(-2.0 * xi * xi * d_sq);
}

__global__ void scale_kernel_joint_polydisperse(
    double* __restrict__ fE_grid,
    double* __restrict__ fG_grid,
    const double* __restrict__ scale_coef,
    const double* __restrict__ scale_coef_Q_imag,
    const double* __restrict__ scale_coef_GP_imag,
    const double* __restrict__ scale_coef_GQ_real,
    const double* __restrict__ Qfactor,
    const double* __restrict__ Qfactor_dot,
    size_t num_voxels)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;

    double sc = scale_coef[v];
    double sc_Q = scale_coef_Q_imag[v];
    double sc_GP = scale_coef_GP_imag[v];
    double sc_GQ = scale_coef_GQ_real[v];

    double S_R = fE_grid[v * 2 + 0];
    double S_I = fE_grid[v * 2 + 1];

    double G_dot_Qdot_R = 0.0;
    double G_dot_Qdot_I = 0.0;
    for (int c = 0; c < 5; ++c) {
        double Qdot_c = Qfactor_dot[v * 5 + c];
        G_dot_Qdot_R += fG_grid[(v * 5 + c) * 2 + 0] * Qdot_c;
        G_dot_Qdot_I += fG_grid[(v * 5 + c) * 2 + 1] * Qdot_c;
    }

    double S_new_R = sc * S_R - sc_Q * G_dot_Qdot_R;
    double S_new_I = sc * S_I - sc_Q * G_dot_Qdot_I;

    double Gdot_G_R = sc_GP * S_R - sc_GQ * G_dot_Qdot_R;
    double Gdot_G_I = sc_GP * S_I - sc_GQ * G_dot_Qdot_I;

    fE_grid[v * 2 + 0] = S_new_R;
    fE_grid[v * 2 + 1] = S_new_I;

    for (int c = 0; c < 5; ++c) {
        double Qf_c = Qfactor[v * 5 + c];
        fG_grid[(v * 5 + c) * 2 + 0] = Qf_c * Gdot_G_R;
        fG_grid[(v * 5 + c) * 2 + 1] = Qf_c * Gdot_G_I;
    }
}

__global__ void spread_quadrupoles_kernel_polydisperse(
    const double* __restrict__ d_dipoles,
    const int* __restrict__ quad_idxs,
    const double* __restrict__ spread_coef,
    const int* __restrict__ spread_idxs,
    double* __restrict__ fG_grid,
    size_t num_quads,
    size_t num_particles,
    size_t num_offsets,
    int num_grid_x,
    int num_grid_y,
    int num_grid_z)
{
    __shared__ int block_min_gx, block_max_gx;
    __shared__ int block_min_gy, block_max_gy;
    __shared__ int block_min_gz, block_max_gz;

    if (threadIdx.x == 0) {
        block_min_gx = 999999; block_max_gx = -999999;
        block_min_gy = 999999; block_max_gy = -999999;
        block_min_gz = 999999; block_max_gz = -999999;
    }
    __syncthreads();

    size_t total_spread_Q = num_quads * num_offsets;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (idx < total_spread_Q);

    int gx = 0, gy = 0, gz = 0;
    double coef = 0.0;
    double q0_r = 0.0, q0_i = 0.0;
    double q1_r = 0.0, q1_i = 0.0;
    double q2_r = 0.0, q2_i = 0.0;
    double q3_r = 0.0, q3_i = 0.0;
    double q4_r = 0.0, q4_i = 0.0;

    if (active) {
        int q = idx / num_offsets;
        int o = idx % num_offsets;

        int p_idx = quad_idxs[q];
        int part_spread_idx = p_idx * num_offsets + o;

        const double2* d_quad_d2 = reinterpret_cast<const double2*>(d_dipoles + num_particles * 3 * 2);
        double2 q0 = d_quad_d2[q * 5 + 0];
        double2 q1 = d_quad_d2[q * 5 + 1];
        double2 q2 = d_quad_d2[q * 5 + 2];
        double2 q3 = d_quad_d2[q * 5 + 3];
        double2 q4 = d_quad_d2[q * 5 + 4];

        q0_r = q0.x; q0_i = q0.y;
        q1_r = q1.x; q1_i = q1.y;
        q2_r = q2.x; q2_i = q2.y;
        q3_r = q3.x; q3_i = q3.y;
        q4_r = q4.x; q4_i = q4.y;

        coef = spread_coef[part_spread_idx];

        uint32_t packed = spread_idxs[part_spread_idx];
        gx = static_cast<int>(packed >> 20) - 256;
        gy = static_cast<int>((packed >> 10) & 0x3FF) - 256;
        gz = static_cast<int>(packed & 0x3FF) - 256;

        atomicMin(&block_min_gx, gx);
        atomicMax(&block_max_gx, gx);
        atomicMin(&block_min_gy, gy);
        atomicMax(&block_max_gy, gy);
        atomicMin(&block_min_gz, gz);
        atomicMax(&block_max_gz, gz);
    }
    __syncthreads();

    int dim_x = block_max_gx - block_min_gx + 1;
    int dim_y = block_max_gy - block_min_gy + 1;
    int dim_z = block_max_gz - block_min_gz + 1;
    int local_grid_size = dim_x * dim_y * dim_z;

    extern __shared__ double s_grid[];

    bool use_shared = (local_grid_size > 0 && local_grid_size <= 300);

    if (use_shared) {
        for (int offset = threadIdx.x; offset < local_grid_size * 10; offset += blockDim.x) {
            s_grid[offset] = 0.0;
        }
        __syncthreads();

        if (active) {
            int local_x = gx - block_min_gx;
            int local_y = gy - block_min_gy;
            int local_z = gz - block_min_gz;
            int local_idx = local_x * dim_y * dim_z + local_y * dim_z + local_z;

            atomicAdd(&s_grid[(local_idx * 5 + 0) * 2 + 0], coef * q0_r);
            atomicAdd(&s_grid[(local_idx * 5 + 0) * 2 + 1], coef * q0_i);
            atomicAdd(&s_grid[(local_idx * 5 + 1) * 2 + 0], coef * q1_r);
            atomicAdd(&s_grid[(local_idx * 5 + 1) * 2 + 1], coef * q1_i);
            atomicAdd(&s_grid[(local_idx * 5 + 2) * 2 + 0], coef * q2_r);
            atomicAdd(&s_grid[(local_idx * 5 + 2) * 2 + 1], coef * q2_i);
            atomicAdd(&s_grid[(local_idx * 5 + 3) * 2 + 0], coef * q3_r);
            atomicAdd(&s_grid[(local_idx * 5 + 3) * 2 + 1], coef * q3_i);
            atomicAdd(&s_grid[(local_idx * 5 + 4) * 2 + 0], coef * q4_r);
            atomicAdd(&s_grid[(local_idx * 5 + 4) * 2 + 1], coef * q4_i);
        }
        __syncthreads();

        for (int offset = threadIdx.x; offset < local_grid_size; offset += blockDim.x) {
            int local_x = offset / (dim_y * dim_z);
            int local_y = (offset / dim_z) % dim_y;
            int local_z = offset % dim_z;

            int global_gx = ((block_min_gx + local_x) % num_grid_x + num_grid_x) % num_grid_x;
            int global_gy = ((block_min_gy + local_y) % num_grid_y + num_grid_y) % num_grid_y;
            int global_gz = ((block_min_gz + local_z) % num_grid_z + num_grid_z) % num_grid_z;

            size_t global_idx = (static_cast<size_t>(global_gx) * num_grid_y * num_grid_z +
                                  static_cast<size_t>(global_gy) * num_grid_z +
                                  static_cast<size_t>(global_gz)) * 5;

            double val_0_r = s_grid[(offset * 5 + 0) * 2 + 0];
            double val_0_i = s_grid[(offset * 5 + 0) * 2 + 1];
            double val_1_r = s_grid[(offset * 5 + 1) * 2 + 0];
            double val_1_i = s_grid[(offset * 5 + 1) * 2 + 1];
            double val_2_r = s_grid[(offset * 5 + 2) * 2 + 0];
            double val_2_i = s_grid[(offset * 5 + 2) * 2 + 1];
            double val_3_r = s_grid[(offset * 5 + 3) * 2 + 0];
            double val_3_i = s_grid[(offset * 5 + 3) * 2 + 1];
            double val_4_r = s_grid[(offset * 5 + 4) * 2 + 0];
            double val_4_i = s_grid[(offset * 5 + 4) * 2 + 1];

            if (val_0_r != 0.0) atomicAdd(&fG_grid[(global_idx + 0) * 2 + 0], val_0_r);
            if (val_0_i != 0.0) atomicAdd(&fG_grid[(global_idx + 0) * 2 + 1], val_0_i);
            if (val_1_r != 0.0) atomicAdd(&fG_grid[(global_idx + 1) * 2 + 0], val_1_r);
            if (val_1_i != 0.0) atomicAdd(&fG_grid[(global_idx + 1) * 2 + 1], val_1_i);
            if (val_2_r != 0.0) atomicAdd(&fG_grid[(global_idx + 2) * 2 + 0], val_2_r);
            if (val_2_i != 0.0) atomicAdd(&fG_grid[(global_idx + 2) * 2 + 1], val_2_i);
            if (val_3_r != 0.0) atomicAdd(&fG_grid[(global_idx + 3) * 2 + 0], val_3_r);
            if (val_3_i != 0.0) atomicAdd(&fG_grid[(global_idx + 3) * 2 + 1], val_3_i);
            if (val_4_r != 0.0) atomicAdd(&fG_grid[(global_idx + 4) * 2 + 0], val_4_r);
            if (val_4_i != 0.0) atomicAdd(&fG_grid[(global_idx + 4) * 2 + 1], val_4_i);
        }
    } else {
        if (active) {
            int global_gx = ((gx) % num_grid_x + num_grid_x) % num_grid_x;
            int global_gy = ((gy) % num_grid_y + num_grid_y) % num_grid_y;
            int global_gz = ((gz) % num_grid_z + num_grid_z) % num_grid_z;

            size_t global_idx = (static_cast<size_t>(global_gx) * num_grid_y * num_grid_z +
                                  static_cast<size_t>(global_gy) * num_grid_z +
                                  static_cast<size_t>(global_gz)) * 5;

            atomicAdd(&fG_grid[(global_idx + 0) * 2 + 0], coef * q0_r);
            atomicAdd(&fG_grid[(global_idx + 0) * 2 + 1], coef * q0_i);
            atomicAdd(&fG_grid[(global_idx + 1) * 2 + 0], coef * q1_r);
            atomicAdd(&fG_grid[(global_idx + 1) * 2 + 1], coef * q1_i);
            atomicAdd(&fG_grid[(global_idx + 2) * 2 + 0], coef * q2_r);
            atomicAdd(&fG_grid[(global_idx + 2) * 2 + 1], coef * q2_i);
            atomicAdd(&fG_grid[(global_idx + 3) * 2 + 0], coef * q3_r);
            atomicAdd(&fG_grid[(global_idx + 3) * 2 + 1], coef * q3_i);
            atomicAdd(&fG_grid[(global_idx + 4) * 2 + 0], coef * q4_r);
            atomicAdd(&fG_grid[(global_idx + 4) * 2 + 1], coef * q4_i);
        }
    }
}

__global__ void contract_kernel_G_polydisperse(
    const double* __restrict__ Gs_grid,
    const int* __restrict__ contract_idxs,
    const double* __restrict__ contract_coef,
    double* __restrict__ G_point,
    size_t num_field_points,
    size_t num_offsets,
    int num_grid_y,
    int num_grid_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_field_points) return;

    double G_0_r = 0.0, G_0_i = 0.0;
    double G_1_r = 0.0, G_1_i = 0.0;
    double G_2_r = 0.0, G_2_i = 0.0;
    double G_3_r = 0.0, G_3_i = 0.0;
    double G_4_r = 0.0, G_4_i = 0.0;

    for (size_t o = 0; o < num_offsets; ++o) {
        size_t idx = o * num_field_points + i;
        int v = contract_idxs[idx];

        double coef = contract_coef[idx];

        G_0_r += coef * Gs_grid[(v * 5 + 0) * 2 + 0];
        G_0_i += coef * Gs_grid[(v * 5 + 0) * 2 + 1];

        G_1_r += coef * Gs_grid[(v * 5 + 1) * 2 + 0];
        G_1_i += coef * Gs_grid[(v * 5 + 1) * 2 + 1];

        G_2_r += coef * Gs_grid[(v * 5 + 2) * 2 + 0];
        G_2_i += coef * Gs_grid[(v * 5 + 2) * 2 + 1];

        G_3_r += coef * Gs_grid[(v * 5 + 3) * 2 + 0];
        G_3_i += coef * Gs_grid[(v * 5 + 3) * 2 + 1];

        G_4_r += coef * Gs_grid[(v * 5 + 4) * 2 + 0];
        G_4_i += coef * Gs_grid[(v * 5 + 4) * 2 + 1];
    }

    G_point[(i * 5 + 0) * 2 + 0] = G_0_r;
    G_point[(i * 5 + 0) * 2 + 1] = G_0_i;
    G_point[(i * 5 + 1) * 2 + 0] = G_1_r;
    G_point[(i * 5 + 1) * 2 + 1] = G_1_i;
    G_point[(i * 5 + 2) * 2 + 0] = G_2_r;
    G_point[(i * 5 + 2) * 2 + 1] = G_2_i;
    G_point[(i * 5 + 3) * 2 + 0] = G_3_r;
    G_point[(i * 5 + 3) * 2 + 1] = G_3_i;
    G_point[(i * 5 + 4) * 2 + 0] = G_4_r;
    G_point[(i * 5 + 4) * 2 + 1] = G_4_i;
}

__global__ void copy_G_to_E_kernel_polydisperse(
    const double* __restrict__ G_point,
    const int* __restrict__ quad_idxs,
    double* __restrict__ E_point,
    size_t num_quads,
    size_t num_field_points)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_quads) return;

    int p_idx = quad_idxs[q];
    if (p_idx < 0 || p_idx >= num_field_points) return;

    double* dst = E_point + (num_field_points * 3 + q * 5) * 2;
    const double* src = G_point + (p_idx * 5) * 2;

    for (int c = 0; c < 10; ++c) {
        dst[c] = src[c];
    }
}

__global__ void fftshift_3d_kernel_polydisperse(
    const double* __restrict__ input,
    double* __restrict__ output,
    int N0, int N1, int N2,
    int num_components)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_voxels = N0 * N1 * N2;
    if (idx >= num_voxels) return;

    int x = idx / (N1 * N2);
    int y = (idx / N2) % N1;
    int z = idx % N2;

    int xs = (x + N0 / 2) % N0;
    int ys = (y + N1 / 2) % N1;
    int zs = (z + N2 / 2) % N2;

    size_t shifted_idx = (static_cast<size_t>(xs) * N1 * N2 +
                          static_cast<size_t>(ys) * N2 +
                          static_cast<size_t>(zs));

    for (int c = 0; c < num_components; ++c) {
        output[(shifted_idx * num_components + c) * 2 + 0] = input[(idx * num_components + c) * 2 + 0];
        output[(shifted_idx * num_components + c) * 2 + 1] = input[(idx * num_components + c) * 2 + 1];
    }
}

__global__ void ifftshift_3d_kernel_polydisperse(
    const double* __restrict__ input,
    double* __restrict__ output,
    int N0, int N1, int N2,
    int num_components)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_voxels = N0 * N1 * N2;
    if (idx >= num_voxels) return;

    int x = idx / (N1 * N2);
    int y = (idx / N2) % N1;
    int z = idx % N2;

    int xs = (x + (N0 + 1) / 2) % N0;
    int ys = (y + (N1 + 1) / 2) % N1;
    int zs = (z + (N2 + 1) / 2) % N2;

    size_t shifted_idx = (static_cast<size_t>(xs) * N1 * N2 +
                          static_cast<size_t>(ys) * N2 +
                          static_cast<size_t>(zs));

    double scale_factor = 1.0 / (static_cast<double>(N0) * N1 * N2);

    for (int c = 0; c < num_components; ++c) {
        output[(shifted_idx * num_components + c) * 2 + 0] = input[(idx * num_components + c) * 2 + 0] * scale_factor;
        output[(shifted_idx * num_components + c) * 2 + 1] = input[(idx * num_components + c) * 2 + 1] * scale_factor;
    }
}

__global__ void real_space_self_kernel_polydisperse_joint(
    const double* __restrict__ d_dipoles,
    const double* __restrict__ d_self_coef_r,
    const double* __restrict__ d_self_coef_i,
    const int* __restrict__ radius_idx,
    const double* __restrict__ d_self_perp_uniq,
    const int* __restrict__ quad_idxs,
    const int* __restrict__ quad_map,
    const double* __restrict__ d_self_G2_uniq,
    const double* __restrict__ d_radii,
    double* __restrict__ E_point,
    size_t num_particles,
    size_t num_quads,
    bool solve_quadrupoles,
    FieldCalcMode mode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    double sc_r = d_self_coef_r[i];
    double sc_i = d_self_coef_i[i];

    int radius_idx_i = radius_idx[i];
    double self_perp_val = d_self_perp_uniq[radius_idx_i];

    double factor_r = sc_r + self_perp_val;
    double factor_i = sc_i;

    const double2* d_dipoles_d2 = reinterpret_cast<const double2*>(d_dipoles);
    double2 dip_x = d_dipoles_d2[i * 3 + 0];
    double2 dip_y = d_dipoles_d2[i * 3 + 1];
    double2 dip_z = d_dipoles_d2[i * 3 + 2];

    double dx_r = dip_x.x; double dx_i = dip_x.y;
    double dy_r = dip_y.x; double dy_i = dip_y.y;
    double dz_r = dip_z.x; double dz_i = dip_z.y;

    atomicAdd(&E_point[(i * 3 + 0) * 2 + 0], factor_r * dx_r - factor_i * dx_i);
    atomicAdd(&E_point[(i * 3 + 0) * 2 + 1], factor_r * dx_i + factor_i * dx_r);

    atomicAdd(&E_point[(i * 3 + 1) * 2 + 0], factor_r * dy_r - factor_i * dy_i);
    atomicAdd(&E_point[(i * 3 + 1) * 2 + 1], factor_r * dy_i + factor_i * dy_r);

    atomicAdd(&E_point[(i * 3 + 2) * 2 + 0], factor_r * dz_r - factor_i * dz_i);
    atomicAdd(&E_point[(i * 3 + 2) * 2 + 1], factor_r * dz_i + factor_i * dz_r);

    if (solve_quadrupoles) {
        int q = quad_map[i];
        if (q >= 0 && q < num_quads) {
            double a_j = d_radii[i];
            double self_G2_val = d_self_G2_uniq[radius_idx_i];
            double q_sc_r = 2.5 * sc_r / (a_j * a_j) + self_G2_val;
            double q_sc_i = 2.5 * sc_i / (a_j * a_j);

            const double2* d_quad_d2 = reinterpret_cast<const double2*>(d_dipoles + num_particles * 3 * 2);
            double2 q0 = d_quad_d2[q * 5 + 0];
            double2 q1 = d_quad_d2[q * 5 + 1];
            double2 q2 = d_quad_d2[q * 5 + 2];
            double2 q3 = d_quad_d2[q * 5 + 3];
            double2 q4 = d_quad_d2[q * 5 + 4];

            double q0_r = q0.x; double q0_i = q0.y;
            double q1_r = q1.x; double q1_i = q1.y;
            double q2_r = q2.x; double q2_i = q2.y;
            double q3_r = q3.x; double q3_i = q3.y;
            double q4_r = q4.x; double q4_i = q4.y;

            double* G_point = E_point + num_particles * 3 * 2;

            atomicAdd(&G_point[(q * 5 + 0) * 2 + 0], q_sc_r * q0_r - q_sc_i * q0_i);
            atomicAdd(&G_point[(q * 5 + 0) * 2 + 1], q_sc_r * q0_i + q_sc_i * q0_r);

            atomicAdd(&G_point[(q * 5 + 1) * 2 + 0], q_sc_r * q1_r - q_sc_i * q1_i);
            atomicAdd(&G_point[(q * 5 + 1) * 2 + 1], q_sc_r * q1_i + q_sc_i * q1_r);

            atomicAdd(&G_point[(q * 5 + 2) * 2 + 0], q_sc_r * q2_r - q_sc_i * q2_i);
            atomicAdd(&G_point[(q * 5 + 2) * 2 + 1], q_sc_r * q2_i + q_sc_i * q2_r);

            atomicAdd(&G_point[(q * 5 + 3) * 2 + 0], q_sc_r * q3_r - q_sc_i * q3_i);
            atomicAdd(&G_point[(q * 5 + 3) * 2 + 1], q_sc_r * q3_i + q_sc_i * q3_r);

            atomicAdd(&G_point[(q * 5 + 4) * 2 + 0], q_sc_r * q4_r - q_sc_i * q4_i);
            atomicAdd(&G_point[(q * 5 + 4) * 2 + 1], q_sc_r * q4_i + q_sc_i * q4_r);
        }
    }
}

__global__ void real_space_neighbor_kernel_polydisperse_joint(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
    const double* __restrict__ d_dipoles,
    const int* __restrict__ neighbor_list,
    const int* __restrict__ neighbor_counts,
    const int* __restrict__ particle_offsets,
    const int* __restrict__ quad_idxs,
    const int* __restrict__ quad_map,
    const double* __restrict__ perp,
    const double* __restrict__ para,
    const double* __restrict__ perp_Q,
    const double* __restrict__ para_Q,
    const double* __restrict__ Q3,
    const double* __restrict__ G1,
    const double* __restrict__ G2,
    const double* __restrict__ G3,
    const double* __restrict__ G4,
    double* __restrict__ E_point,
    size_t num_particles,
    size_t num_quads,
    int max_neighbors,
    double box_x,
    double box_y,
    double box_z,
    double rc,
    bool solve_quadrupoles,
    FieldCalcMode mode)
{
    double sign = (mode == FieldCalcMode::SOLVER_AX) ? 1.0 : -1.0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    int count = neighbor_counts[i];
    if (count > max_neighbors) count = max_neighbors;
    int start_idx = particle_offsets[i];

    double xi = x_part[i];
    double yi = y_part[i];
    double zi = z_part[i];

    const double2* d_dipoles_d2 = reinterpret_cast<const double2*>(d_dipoles);
    double2 dip_x = d_dipoles_d2[i * 3 + 0];
    double2 dip_y = d_dipoles_d2[i * 3 + 1];
    double2 dip_z = d_dipoles_d2[i * 3 + 2];

    double px_r = dip_x.x; double px_i = dip_x.y;
    double py_r = dip_y.x; double py_i = dip_y.y;
    double pz_r = dip_z.x; double pz_i = dip_z.y;

    int q_src = -1;
    double q0_r = 0.0, q0_i = 0.0;
    double q1_r = 0.0, q1_i = 0.0;
    double q2_r = 0.0, q2_i = 0.0;
    double q3_r = 0.0, q3_i = 0.0;
    double q4_r = 0.0, q4_i = 0.0;

    if (solve_quadrupoles) {
        q_src = quad_map[i];
        if (q_src >= 0 && q_src < num_quads) {
            const double2* d_quad_d2 = reinterpret_cast<const double2*>(d_dipoles + num_particles * 3 * 2);
            double2 q0 = d_quad_d2[q_src * 5 + 0];
            double2 q1 = d_quad_d2[q_src * 5 + 1];
            double2 q2 = d_quad_d2[q_src * 5 + 2];
            double2 q3 = d_quad_d2[q_src * 5 + 3];
            double2 q4 = d_quad_d2[q_src * 5 + 4];

            q0_r = q0.x; q0_i = q0.y;
            q1_r = q1.x; q1_i = q1.y;
            q2_r = q2.x; q2_i = q2.y;
            q3_r = q3.x; q3_i = q3.y;
            q4_r = q4.x; q4_i = q4.y;
        }
    }

    double* G_point = E_point + num_particles * 3 * 2;
    const double I_arr[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    for (int k = 0; k < count; ++k) {
        int j = neighbor_list[i * max_neighbors + k];

        double rx = x_field[j] - xi;
        double ry = y_field[j] - yi;
        double rz = z_field[j] - zi;

        if (box_x > 0.0) rx -= box_x * round(rx / box_x);
        if (box_y > 0.0) ry -= box_y * round(ry / box_y);
        if (box_z > 0.0) rz -= box_z * round(rz / box_z);

        double d = sqrt(rx * rx + ry * ry + rz * rz);

        if (d < rc) {
            double d_eff = d < 1.0 ? 1.0 : d;
            double delta_x = rx / d_eff;
            double delta_y = ry / d_eff;
            double delta_z = rz / d_eff;

            double __rr[5];
            __rr[0] = delta_x * delta_x - delta_z * delta_z;
            __rr[1] = 2.0 * delta_x * delta_y;
            __rr[2] = 2.0 * delta_x * delta_z;
            __rr[3] = delta_y * delta_y - delta_z * delta_z;
            __rr[4] = 2.0 * delta_y * delta_z;

            // 1. Dipole contributions to E
            double r_P_r = px_r * delta_x + py_r * delta_y + pz_r * delta_z;
            double r_P_i = px_i * delta_x + py_i * delta_y + pz_i * delta_z;

            double perp_val = perp[start_idx + k];
            double para_val = para[start_idx + k];

            double E_dip_x_r = perp_val * (px_r - delta_x * r_P_r) + para_val * delta_x * r_P_r;
            double E_dip_x_i = perp_val * (px_i - delta_x * r_P_i) + para_val * delta_x * r_P_i;

            double E_dip_y_r = perp_val * (py_r - delta_y * r_P_r) + para_val * delta_y * r_P_r;
            double E_dip_y_i = perp_val * (py_i - delta_y * r_P_i) + para_val * delta_y * r_P_i;

            double E_dip_z_r = perp_val * (pz_r - delta_z * r_P_r) + para_val * delta_z * r_P_r;
            double E_dip_z_i = perp_val * (pz_i - delta_z * r_P_i) + para_val * delta_z * r_P_i;

            double E_quad_x_r = 0.0, E_quad_x_i = 0.0;
            double E_quad_y_r = 0.0, E_quad_y_i = 0.0;
            double E_quad_z_r = 0.0, E_quad_z_i = 0.0;

            if (solve_quadrupoles && q_src >= 0) {
                // Quadrupole contributions to E
                double Q_r_vec_r[3];
                Q_r_vec_r[0] = q0_r * delta_x + q1_r * delta_y + q2_r * delta_z;
                Q_r_vec_r[1] = q1_r * delta_x + q3_r * delta_y + q4_r * delta_z;
                Q_r_vec_r[2] = q2_r * delta_x + q4_r * delta_y - (q0_r + q3_r) * delta_z;

                double Q_r_vec_i[3];
                Q_r_vec_i[0] = q0_i * delta_x + q1_i * delta_y + q2_i * delta_z;
                Q_r_vec_i[1] = q1_i * delta_x + q3_i * delta_y + q4_i * delta_z;
                Q_r_vec_i[2] = q2_i * delta_x + q4_i * delta_y - (q0_i + q3_i) * delta_z;

                double Q__rr_r = q0_r * __rr[0] + q1_r * __rr[1] + q2_r * __rr[2] + q3_r * __rr[3] + q4_r * __rr[4];
                double Q__rr_i = q0_i * __rr[0] + q1_i * __rr[1] + q2_i * __rr[2] + q3_i * __rr[3] + q4_i * __rr[4];

                double Q1_val = perp_Q[start_idx + k];
                double Q2_val = para_Q[start_idx + k];

                E_quad_x_r = 0.5 * (Q1_val * Q__rr_r * delta_x + 2.0 * Q2_val * Q_r_vec_r[0]);
                E_quad_x_i = 0.5 * (Q1_val * Q__rr_i * delta_x + 2.0 * Q2_val * Q_r_vec_i[0]);

                E_quad_y_r = 0.5 * (Q1_val * Q__rr_r * delta_y + 2.0 * Q2_val * Q_r_vec_r[1]);
                E_quad_y_i = 0.5 * (Q1_val * Q__rr_i * delta_y + 2.0 * Q2_val * Q_r_vec_i[1]);

                E_quad_z_r = 0.5 * (Q1_val * Q__rr_r * delta_z + 2.0 * Q2_val * Q_r_vec_r[2]);
                E_quad_z_i = 0.5 * (Q1_val * Q__rr_i * delta_z + 2.0 * Q2_val * Q_r_vec_i[2]);
            }

            atomicAdd(&E_point[(j * 3 + 0) * 2 + 0], sign * (E_dip_x_r + E_quad_x_r));
            atomicAdd(&E_point[(j * 3 + 0) * 2 + 1], sign * (E_dip_x_i + E_quad_x_i));

            atomicAdd(&E_point[(j * 3 + 1) * 2 + 0], sign * (E_dip_y_r + E_quad_y_r));
            atomicAdd(&E_point[(j * 3 + 1) * 2 + 1], sign * (E_dip_y_i + E_quad_y_i));

            atomicAdd(&E_point[(j * 3 + 2) * 2 + 0], sign * (E_dip_z_r + E_quad_z_r));
            atomicAdd(&E_point[(j * 3 + 2) * 2 + 1], sign * (E_dip_z_i + E_quad_z_i));

            // 2. Contributions to G
            if (solve_quadrupoles) {
                int q_field = (j < num_particles) ? quad_map[j] : -1;
                if (q_field >= 0 && q_field < num_quads) {
                    double Q1_val = perp_Q[start_idx + k];
                    double Q2_val = para_Q[start_idx + k];
                    double Q3_val = Q3[start_idx + k];

                    double Pr_rP_r[5];
                    Pr_rP_r[0] = 2.0 * (px_r * delta_x - pz_r * delta_z);
                    Pr_rP_r[1] = 2.0 * (px_r * delta_y + py_r * delta_x);
                    Pr_rP_r[2] = 2.0 * (px_r * delta_z + pz_r * delta_x);
                    Pr_rP_r[3] = 2.0 * (py_r * delta_y - pz_r * delta_z);
                    Pr_rP_r[4] = 2.0 * (py_r * delta_z + pz_r * delta_y);

                    double Pr_rP_i[5];
                    Pr_rP_i[0] = 2.0 * (px_i * delta_x - pz_i * delta_z);
                    Pr_rP_i[1] = 2.0 * (px_i * delta_y + py_i * delta_x);
                    Pr_rP_i[2] = 2.0 * (px_i * delta_z + pz_i * delta_x);
                    Pr_rP_i[3] = 2.0 * (py_i * delta_y - pz_i * delta_z);
                    Pr_rP_i[4] = 2.0 * (py_i * delta_z + pz_i * delta_y);

                    double G_dip_r[5];
                    double G_dip_i[5];
                    for (int c = 0; c < 5; ++c) {
                        G_dip_r[c] = -(Q1_val * __rr[c] * r_P_r + Q2_val * Pr_rP_r[c] + (Q2_val + Q3_val) * I_arr[c] * r_P_r);
                        G_dip_i[c] = -(Q1_val * __rr[c] * r_P_i + Q2_val * Pr_rP_i[c] + (Q2_val + Q3_val) * I_arr[c] * r_P_i);
                    }

                    double G_quad_r[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
                    double G_quad_i[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

                    if (q_src >= 0) {
                        double G1_val = G1[start_idx + k];
                        double G2_val = G2[start_idx + k];
                        double G3_val = G3[start_idx + k];
                        double G4_val = G4[start_idx + k];

                        double Q_r_vec_r[3];
                        Q_r_vec_r[0] = q0_r * delta_x + q1_r * delta_y + q2_r * delta_z;
                        Q_r_vec_r[1] = q1_r * delta_x + q3_r * delta_y + q4_r * delta_z;
                        Q_r_vec_r[2] = q2_r * delta_x + q4_r * delta_y - (q0_r + q3_r) * delta_z;

                        double Q_r_vec_i[3];
                        Q_r_vec_i[0] = q0_i * delta_x + q1_i * delta_y + q2_i * delta_z;
                        Q_r_vec_i[1] = q1_i * delta_x + q3_i * delta_y + q4_i * delta_z;
                        Q_r_vec_i[2] = q2_i * delta_x + q4_i * delta_y - (q0_i + q3_i) * delta_z;

                        double Q__rr_r = q0_r * __rr[0] + q1_r * __rr[1] + q2_r * __rr[2] + q3_r * __rr[3] + q4_r * __rr[4];
                        double Q__rr_i = q0_i * __rr[0] + q1_i * __rr[1] + q2_i * __rr[2] + q3_i * __rr[3] + q4_i * __rr[4];

                        double Q_rr_rr_Q_r[5];
                        Q_rr_rr_Q_r[0] = 2.0 * (Q_r_vec_r[0] * delta_x - Q_r_vec_r[2] * delta_z);
                        Q_rr_rr_Q_r[1] = 2.0 * (Q_r_vec_r[0] * delta_y + Q_r_vec_r[1] * delta_x);
                        Q_rr_rr_Q_r[2] = 2.0 * (Q_r_vec_r[0] * delta_z + Q_r_vec_r[2] * delta_x);
                        Q_rr_rr_Q_r[3] = 2.0 * (Q_r_vec_r[1] * delta_y - Q_r_vec_r[2] * delta_z);
                        Q_rr_rr_Q_r[4] = 2.0 * (Q_r_vec_r[1] * delta_z + Q_r_vec_r[2] * delta_y);

                        double Q_rr_rr_Q_i[5];
                        Q_rr_rr_Q_i[0] = 2.0 * (Q_r_vec_i[0] * delta_x - Q_r_vec_i[2] * delta_z);
                        Q_rr_rr_Q_i[1] = 2.0 * (Q_r_vec_i[0] * delta_y + Q_r_vec_i[1] * delta_x);
                        Q_rr_rr_Q_i[2] = 2.0 * (Q_r_vec_i[0] * delta_z + Q_r_vec_i[2] * delta_x);
                        Q_rr_rr_Q_i[3] = 2.0 * (Q_r_vec_i[1] * delta_y - Q_r_vec_i[2] * delta_z);
                        Q_rr_rr_Q_i[4] = 2.0 * (Q_r_vec_i[1] * delta_z + Q_r_vec_i[2] * delta_y);

                        double q_val_r[5] = {q0_r, q1_r, q2_r, q3_r, q4_r};
                        double q_val_i[5] = {q0_i, q1_i, q2_i, q3_i, q4_i};

                        for (int c = 0; c < 5; ++c) {
                            G_quad_r[c] = 0.5 * (G1_val * I_arr[c] * Q__rr_r + G2_val * q_val_r[c] + G3_val * (I_arr[c] * Q__rr_r + 2.0 * Q_rr_rr_Q_r[c]) + G4_val * __rr[c] * Q__rr_r);
                            G_quad_i[c] = 0.5 * (G1_val * I_arr[c] * Q__rr_i + G2_val * q_val_i[c] + G3_val * (I_arr[c] * Q__rr_i + 2.0 * Q_rr_rr_Q_i[c]) + G4_val * __rr[c] * Q__rr_i);
                        }
                    }

                    for (int c = 0; c < 5; ++c) {
                        atomicAdd(&G_point[(q_field * 5 + c) * 2 + 0], sign * (G_dip_r[c] + G_quad_r[c]));
                        atomicAdd(&G_point[(q_field * 5 + c) * 2 + 1], sign * (G_dip_i[c] + G_quad_i[c]));
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Polydisperse_Ewald_Electric_Field Implementation
// -----------------------------------------------------------------------------

static void helper_compute_pair_tables(double a_i, double a_j, double xi,
                                       const std::vector<double>& r_vals,
                                       std::vector<double>& fd1,
                                       std::vector<double>& fd2,
                                       double& self_perp_val)
{
    const double PI = 3.14159265358979323846;
    double pi_pow_1_5 = std::pow(PI, 1.5);
    double xi_sq = xi * xi;
    double xi_cub = xi_sq * xi;
    double xi_4 = xi_sq * xi_sq;
    double xi_5 = xi_4 * xi;
    double xi_6 = xi_5 * xi;

    size_t num_r_steps = r_vals.size();

    for (size_t idx = 0; idx < num_r_steps; ++idx) {
        double r = r_vals[idx];
        double r_sq = r * r;
        double r_cub = r_sq * r;
        double r_4 = r_sq * r_sq;
        double r_5 = r_4 * r;
        double r_6 = r_4 * r_sq;

        double f_1 = 1.0 / (1024.0 * pi_pow_1_5 * std::pow(a_i, 3.0) * std::pow(a_j, 3.0) * r_cub * xi_5) * 
            (4.0 * r_5 * xi_4 - 4.0 * (a_i + a_j) * r_4 * xi_4 + 
             r_cub * (16.0 * xi_sq + 8.0 * (-4.0 * a_i * a_i + a_i * a_j - 4.0 * a_j * a_j) * xi_4) + 
             r_sq * (-12.0 * (a_i + a_j) * xi_sq - 8.0 * (4.0 * std::pow(a_i, 3) - 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j + 4.0 * std::pow(a_j, 3)) * xi_4) + 
             r * (3.0 - 12.0 * (a_i * a_i - a_i * a_j + a_j * a_j) * xi_sq - 4.0 * std::pow(a_i + a_j, 2) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_4) + 
             3.0 * (a_i + a_j) + 4.0 * (4.0 * std::pow(a_i, 3) - 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j + 4.0 * std::pow(a_j, 3)) * xi_sq + 
             4.0 * std::pow(a_i + a_j, 3) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_4);

        double f_2 = 1.0 / (1024.0 * pi_pow_1_5 * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_5) * 
            (4.0 * r_5 * xi_4 + 4.0 * (a_i + a_j) * r_4 * xi_4 + 
             r_cub * (16.0 * xi_sq + 8.0 * (-4.0 * a_i * a_i + a_i * a_j - 4.0 * a_j * a_j) * xi_4) + 
             r_sq * (12.0 * (a_i + a_j) * xi_sq + 8.0 * (4.0 * std::pow(a_i, 3) - 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j + 4.0 * std::pow(a_j, 3)) * xi_4) + 
             r * (3.0 - 12.0 * (a_i * a_i - a_i * a_j + a_j * a_j) * xi_sq - 4.0 * std::pow(a_i + a_j, 2) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_4) - 
             3.0 * (a_i + a_j) - 4.0 * (4.0 * std::pow(a_i, 3) - 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j + 4.0 * std::pow(a_j, 3)) * xi_sq - 
             4.0 * std::pow(a_i + a_j, 3) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_4);

        double f_3 = 1.0 / (1024.0 * pi_pow_1_5 * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_5) * 
            (-4.0 * r_5 * xi_4 + 4.0 * (-a_i + a_j) * r_4 * xi_4 + 
             r_cub * (-16.0 * xi_sq + 8.0 * (4.0 * a_i * a_i + a_i * a_j + 4.0 * a_j * a_j) * xi_4) + 
             r_sq * (12.0 * (-a_i + a_j) * xi_sq - 8.0 * (4.0 * std::pow(a_i, 3) + 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j - std::pow(a_j, 3)) * xi_4) + 
             r * (-3.0 + 12.0 * (a_i * a_i + a_i * a_j + a_j * a_j) * xi_sq + 4.0 * std::pow(a_i - a_j, 2) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_4) + 
             3.0 * (a_i - a_j) + 4.0 * (4.0 * std::pow(a_i, 3) + 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j - 4.0 * std::pow(a_j, 3)) * xi_sq + 
             4.0 * std::pow(a_i - a_j, 3) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_4);

        double f_4 = 1.0 / (1024.0 * pi_pow_1_5 * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_5) * 
            (-4.0 * r_5 * xi_4 + 4.0 * (a_i - a_j) * r_4 * xi_4 + 
             r_cub * (-16.0 * xi_sq + 8.0 * (4.0 * a_i * a_i + a_i * a_j + 4.0 * a_j * a_j) * xi_4) + 
             r_sq * (12.0 * (a_i - a_j) * xi_sq + 8.0 * (4.0 * std::pow(a_i, 3) + 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j - std::pow(a_j, 3)) * xi_4) + 
             r * (-3.0 + 12.0 * (a_i * a_i + a_i * a_j + a_j * a_j) * xi_sq + 4.0 * std::pow(a_i - a_j, 2) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_4) + 
             3.0 * (-a_i + a_j) - 4.0 * (4.0 * std::pow(a_i, 3) + 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j - 4.0 * std::pow(a_j, 3)) * xi_sq - 
             4.0 * std::pow(a_i - a_j, 3) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_4);

        double f_5 = 1.0 / (2048.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_6) * 
            (-8.0 * r_6 * xi_6 + 36.0 * r_4 * xi_4 * (-1.0 + 2.0 * (a_i * a_i + a_j * a_j) * xi_sq) + 
             128.0 * (std::pow(a_i, 3) + std::pow(a_j, 3)) * r_cub * xi_6 + 
             r_sq * (-18.0 * xi_sq + 72.0 * (a_i * a_i + a_j * a_j) * xi_4 + 72.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_6) + 
             3.0 - 18.0 * (a_i * a_i + a_j * a_j) * xi_sq - 36.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_4 - 
             8.0 * std::pow(a_i + a_j, 4) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_6);

        double f_6 = 1.0 / (2048.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_6) * 
            (-8.0 * r_6 * xi_6 + 36.0 * r_4 * xi_4 * (-1.0 + 2.0 * (a_i * a_i + a_j * a_j) * xi_sq) - 
             128.0 * (std::pow(a_i, 3) + std::pow(a_j, 3)) * r_cub * xi_6 + 
             r_sq * (-18.0 * xi_sq + 72.0 * (a_i * a_i + a_j * a_j) * xi_4 + 72.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_6) + 
             3.0 - 18.0 * (a_i * a_i + a_j * a_j) * xi_sq - 36.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_4 - 
             8.0 * std::pow(a_i + a_j, 4) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_6);

        double f_7 = 1.0 / (2048.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_6) * 
            (8.0 * r_6 * xi_6 + 36.0 * r_4 * xi_4 * (1.0 - 2.0 * (a_i * a_i + a_j * a_j) * xi_sq) + 
             128.0 * (std::pow(a_i, 3) - std::pow(a_j, 3)) * r_cub * xi_6 + 
             r_sq * (18.0 * xi_sq - 72.0 * (a_i * a_i + a_j * a_j) * xi_4 - 72.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_6) - 
             3.0 + 18.0 * (a_i * a_i + a_j * a_j) * xi_sq + 36.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_4 + 
             8.0 * std::pow(a_i - a_j, 4) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_6);

        double f_8 = 1.0 / (2048.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_6) * 
            (8.0 * r_6 * xi_6 + 36.0 * r_4 * xi_4 * (1.0 - 2.0 * (a_i * a_i + a_j * a_j) * xi_sq) - 
             128.0 * (std::pow(a_i, 3) - std::pow(a_j, 3)) * r_cub * xi_6 + 
             r_sq * (18.0 * xi_sq - 72.0 * (a_i * a_i + a_j * a_j) * xi_4 - 72.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_6) - 
             3.0 + 18.0 * (a_i * a_i + a_j * a_j) * xi_sq + 36.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_4 + 
             8.0 * std::pow(a_i - a_j, 4) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_6);

        double f_reg = 0.0;
        if (r < a_i + a_j && r >= std::abs(a_i - a_j)) {
            f_reg = std::pow(a_i + a_j - r, 3.0) / (128.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub) * 
                (-r_cub - 3.0 * (a_i + a_j) * r_sq + 3.0 * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * r + 
                 std::pow(a_i, 3) - 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j + std::pow(a_j, 3));
        } else if (r < a_j - a_i) {
            f_reg = (r_cub - std::pow(a_j, 3.0)) / (4.0 * PI * std::pow(a_j, 3.0) * r_cub);
        } else if (r < a_i - a_j) {
            f_reg = (r_cub - std::pow(a_i, 3.0)) / (4.0 * PI * std::pow(a_i, 3.0) * r_cub);
        }

        fd1[idx] = f_1 * std::exp(-std::pow(r + a_i + a_j, 2) * xi_sq) + 
                   f_2 * std::exp(-std::pow(r - a_i - a_j, 2) * xi_sq) + 
                   f_3 * std::exp(-std::pow(r - a_i + a_j, 2) * xi_sq) + 
                   f_4 * std::exp(-std::pow(r + a_i - a_j, 2) * xi_sq) + 
                   f_5 * std::erfc((r + a_i + a_j) * xi) + 
                   f_6 * std::erfc((r - a_i - a_j) * xi) + 
                   f_7 * std::erfc((r - a_i + a_j) * xi) + 
                   f_8 * std::erfc((r + a_i - a_j) * xi) + f_reg;

        double fd2_f_1 = 1.0 / (512.0 * pi_pow_1_5 * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_5) * 
            (8.0 * r_5 * xi_4 - 8.0 * (a_i + a_j) * r_4 * xi_4 + 
             r_cub * (14.0 * xi_sq - 4.0 * (7.0 * a_i * a_i - 4.0 * a_i * a_j + 7.0 * a_j * a_j) * xi_4) + 
             r_sq * (-6.0 * (a_i + a_j) * xi_sq - 4.0 * (std::pow(a_i, 3) - 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j + std::pow(a_j, 3)) * xi_4) + 
             r * (-3.0 + 12.0 * (a_i * a_i - a_i * a_j + a_j * a_j) * xi_sq + 4.0 * std::pow(a_i + a_j, 2) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_4) - 
             3.0 * (a_i + a_j) - 4.0 * (4.0 * std::pow(a_i, 3) - 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j + 4.0 * std::pow(a_j, 3)) * xi_sq - 
             4.0 * std::pow(a_i + a_j, 3) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_4);

        double fd2_f_2 = 1.0 / (512.0 * pi_pow_1_5 * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_5) * 
            (8.0 * r_5 * xi_4 + 8.0 * (a_i + a_j) * r_4 * xi_4 + 
             r_cub * (14.0 * xi_sq - 4.0 * (7.0 * a_i * a_i - 4.0 * a_i * a_j + 7.0 * a_j * a_j) * xi_4) + 
             r_sq * (6.0 * (a_i + a_j) * xi_sq + 4.0 * (std::pow(a_i, 3) - 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j + std::pow(a_j, 3)) * xi_4) + 
             r * (-3.0 + 12.0 * (a_i * a_i - a_i * a_j + a_j * a_j) * xi_sq + 4.0 * std::pow(a_i + a_j, 2) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_4) + 
             3.0 * (a_i + a_j) + 4.0 * (4.0 * std::pow(a_i, 3) - 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j + 4.0 * std::pow(a_j, 3)) * xi_sq + 
             4.0 * std::pow(a_i + a_j, 3) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_4);

        double fd2_f_3 = 1.0 / (512.0 * pi_pow_1_5 * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_5) * 
            (-8.0 * r_5 * xi_4 + 8.0 * (-a_i + a_j) * r_4 * xi_4 + 
             r_cub * (-14.0 * xi_sq + 4.0 * (7.0 * a_i * a_i + 4.0 * a_i * a_j + 7.0 * a_j * a_j) * xi_4) + 
             r_sq * (6.0 * (-a_i + a_j) * xi_sq - 4.0 * (std::pow(a_i, 3) + 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j - std::pow(a_j, 3)) * xi_4) + 
             r * (3.0 - 12.0 * (a_i * a_i + a_i * a_j + a_j * a_j) * xi_sq - 4.0 * std::pow(a_i - a_j, 2) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_4) + 
             3.0 * (-a_i + a_j) - 4.0 * (4.0 * std::pow(a_i, 3) + 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j - 4.0 * std::pow(a_j, 3)) * xi_sq - 
             4.0 * std::pow(a_i - a_j, 3) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_4);

        double fd2_f_4 = 1.0 / (512.0 * pi_pow_1_5 * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_5) * 
            (-8.0 * r_5 * xi_4 + 8.0 * (a_i - a_j) * r_4 * xi_4 + 
             r_cub * (-14.0 * xi_sq + 4.0 * (7.0 * a_i * a_i + 4.0 * a_i * a_j + 7.0 * a_j * a_j) * xi_4) + 
             r_sq * (6.0 * (a_i - a_j) * xi_sq + 4.0 * (std::pow(a_i, 3) + 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j - std::pow(a_j, 3)) * xi_4) + 
             r * (3.0 - 12.0 * (a_i * a_i + a_i * a_j + a_j * a_j) * xi_sq + 4.0 * std::pow(a_i - a_j, 2) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_4) + 
             3.0 * (a_i - a_j) + 4.0 * (4.0 * std::pow(a_i, 3) + 3.0 * a_i * a_i * a_j - 3.0 * a_i * a_j * a_j - 4.0 * std::pow(a_j, 3)) * xi_sq + 
             4.0 * std::pow(a_i - a_j, 3) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_4);

        double fd2_f_5 = 1.0 / (1024.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_6) * 
            (-16.0 * r_6 * xi_6 + 36.0 * r_4 * xi_4 * (-1.0 + 2.0 * (a_i * a_i + a_j * a_j) * xi_sq) + 
             64.0 * (std::pow(a_i, 3) + std::pow(a_j, 3)) * r_cub * xi_6 - 
             3.0 + 18.0 * (a_i * a_i + a_j * a_j) * xi_sq + 36.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_4 + 
             8.0 * std::pow(a_i + a_j, 4) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_6);

        double fd2_f_6 = 1.0 / (1024.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_6) * 
            (-16.0 * r_6 * xi_6 + 36.0 * r_4 * xi_4 * (-1.0 + 2.0 * (a_i * a_i + a_j * a_j) * xi_sq) - 
             64.0 * (std::pow(a_i, 3) + std::pow(a_j, 3)) * r_cub * xi_6 - 
             3.0 + 18.0 * (a_i * a_i + a_j * a_j) * xi_sq + 36.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_4 + 
             8.0 * std::pow(a_i + a_j, 4) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) * xi_6);

        double fd2_f_7 = 1.0 / (1024.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_6) * 
            (16.0 * r_6 * xi_6 + 36.0 * r_4 * xi_4 * (1.0 - 2.0 * (a_i * a_i + a_j * a_j) * xi_sq) + 
             64.0 * (std::pow(a_i, 3) - std::pow(a_j, 3)) * r_cub * xi_6 + 
             3.0 - 18.0 * (a_i * a_i + a_j * a_j) * xi_sq - 36.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_4 - 
             8.0 * std::pow(a_i - a_j, 4) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_6);

        double fd2_f_8 = 1.0 / (1024.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub * xi_6) * 
            (16.0 * r_6 * xi_6 + 36.0 * r_4 * xi_4 * (1.0 - 2.0 * (a_i * a_i + a_j * a_j) * xi_sq) + 
             64.0 * (-std::pow(a_i, 3) + std::pow(a_j, 3)) * r_cub * xi_6 + 
             3.0 - 18.0 * (a_i * a_i + a_j * a_j) * xi_sq - 36.0 * std::pow(a_i * a_i - a_j * a_j, 2) * xi_4 - 
             8.0 * std::pow(a_i - a_j, 4) * (a_i * a_i + 4.0 * a_i * a_j + a_j * a_j) * xi_6);

        double fd2_reg = 0.0;
        if (r < a_i + a_j && r >= std::abs(a_i - a_j)) {
            fd2_reg = -1.0 / (64.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3) * r_cub) * 
                (std::pow(a_i + a_j, 4) * (a_i * a_i - 4.0 * a_i * a_j + a_j * a_j) - 
                 8.0 * (std::pow(a_i, 3) + std::pow(a_j, 3)) * r_cub + 9.0 * (a_i * a_i + a_j * a_j) * r_4 - 2.0 * r_6);
        } else if (r < a_j - a_i) {
            fd2_reg = (r_cub + 2.0 * std::pow(a_j, 3.0)) / (4.0 * PI * std::pow(a_j, 3.0) * r_cub);
        } else if (r < a_i - a_j) {
            fd2_reg = (r_cub + 2.0 * std::pow(a_i, 3.0)) / (4.0 * PI * std::pow(a_i, 3.0) * r_cub);
        }

        fd2[idx] = fd2_f_1 * std::exp(-std::pow(r + a_i + a_j, 2) * xi_sq) + 
                   fd2_f_2 * std::exp(-std::pow(r - a_i - a_j, 2) * xi_sq) + 
                   fd2_f_3 * std::exp(-std::pow(r - a_i + a_j, 2) * xi_sq) + 
                   fd2_f_4 * std::exp(-std::pow(r + a_i - a_j, 2) * xi_sq) + 
                   fd2_f_5 * std::erfc((r + a_i + a_j) * xi) + 
                   fd2_f_6 * std::erfc((r - a_i - a_j) * xi) + 
                   fd2_f_7 * std::erfc((r - a_i + a_j) * xi) + 
                   fd2_f_8 * std::erfc((r + a_i - a_j) * xi) + fd2_reg;
    }

    self_perp_val = 1.0 / (16.0 * pi_pow_1_5 * std::pow(a_i, 3.0) * std::pow(a_j, 3.0) * xi_cub) * 
        ((1.0 - 2.0 * a_i * a_i * xi_sq + 2.0 * a_i * a_j * xi_sq - 2.0 * a_j * a_j * xi_sq) * std::exp(-std::pow(a_i + a_j, 2) * xi_sq) + 
         (-1.0 + 2.0 * a_i * a_i * xi_sq + 2.0 * a_i * a_j * xi_sq + 2.0 * a_j * a_j * xi_sq) * std::exp(-std::pow(a_i - a_j, 2) * xi_sq)) + 
        1.0 / (8.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3)) * 
        ((std::pow(a_i, 3) - std::pow(a_j, 3)) * std::erf((a_i - a_j) * xi) - 
         (std::pow(a_i, 3) + std::pow(a_j, 3)) * std::erf((a_i + a_j) * xi)) + 
        std::min(std::pow(a_i, 3), std::pow(a_j, 3)) / (4.0 * PI * std::pow(a_i, 3) * std::pow(a_j, 3));
}







void Polydisperse_Ewald_Electric_Field::computeRealSpaceTables() {
    size_t num_r_steps = 9000;
    std::vector<double> r_vals(num_r_steps);
    for (size_t idx = 0; idx < num_r_steps; ++idx) {
        r_vals[idx] = 1.0 + idx * 0.001;
    }

    table_size = 9001;
    size_t size_in_bytes = table_size * num_pairs_unique * sizeof(double);

    std::vector<double> host_r_table(table_size);
    host_r_table[0] = 0.0;
    for (size_t idx = 0; idx < num_r_steps; ++idx) {
        host_r_table[idx + 1] = r_vals[idx];
    }

    std::vector<double> host_field_dip_1(table_size * num_pairs_unique, 0.0);
    std::vector<double> host_field_dip_2(table_size * num_pairs_unique, 0.0);
    std::vector<double> host_self_perp_uniq(num_unique_radii, 0.0);

    const double PI = 3.14159265358979323846;
    double pi_pow_1_5 = std::pow(PI, 1.5);
    double xi_sq = xi * xi;
    double xi_cub = xi_sq * xi;
    double xi_4 = xi_sq * xi_sq;
    double xi_5 = xi_4 * xi;
    double xi_6 = xi_5 * xi;

    int lin = 0;
    for (size_t i = 0; i < num_unique_radii; ++i) {
        for (size_t j = i; j < num_unique_radii; ++j) {
            std::vector<double> fd1(num_r_steps);
            std::vector<double> fd2(num_r_steps);
            double self_perp_val = 0.0;

            helper_compute_pair_tables(unique_radii[i], unique_radii[j], xi, r_vals, fd1, fd2, self_perp_val);

            // Store in host tables
            host_field_dip_1[0 * num_pairs_unique + lin] = self_perp_val;
            host_field_dip_2[0 * num_pairs_unique + lin] = self_perp_val;

            for (size_t idx = 0; idx < num_r_steps; ++idx) {
                host_field_dip_1[(idx + 1) * num_pairs_unique + lin] = fd1[idx];
                host_field_dip_2[(idx + 1) * num_pairs_unique + lin] = fd2[idx];
            }

            if (i == j) {
                host_self_perp_uniq[i] = self_perp_val;
            }

            lin++;
        }
    }

    if (d_r_table) CUDA_CHECK(cudaFree(d_r_table));
    if (d_field_dip_1) CUDA_CHECK(cudaFree(d_field_dip_1));
    if (d_field_dip_2) CUDA_CHECK(cudaFree(d_field_dip_2));
    if (d_self_perp_uniq) CUDA_CHECK(cudaFree(d_self_perp_uniq));

    CUDA_CHECK(cudaMalloc(&d_r_table, table_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_field_dip_1, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_field_dip_2, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_self_perp_uniq, num_unique_radii * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_r_table, host_r_table.data(), table_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field_dip_1, host_field_dip_1.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field_dip_2, host_field_dip_2.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_perp_uniq, host_self_perp_uniq.data(), num_unique_radii * sizeof(double), cudaMemcpyHostToDevice));

    if (solve_quadrupoles) {
        if (d_self_G2_uniq) CUDA_CHECK(cudaFree(d_self_G2_uniq));
        CUDA_CHECK(cudaMalloc(&d_self_G2_uniq, num_unique_radii * sizeof(double)));

        std::vector<double> host_self_G2_uniq(num_unique_radii, 0.0);
        for (size_t i = 0; i < num_unique_radii; ++i) {
            double a_j = unique_radii[i];
            double xi_a = xi * a_j;
            double xi_a_sq = xi_a * xi_a;
            double xi_a_4 = xi_a_sq * xi_a_sq;
            double xi_a_5 = xi_a_4 * xi_a;

            double val_1 = 1.0 / (2.0 * PI) * (
                3.0 * (3.0 - 10.0 * xi_a_sq + 20.0 * xi_a_4) / (8.0 * std::sqrt(PI) * xi_a_5) -
                3.0 * (3.0 + 2.0 * xi_a_sq + 4.0 * xi_a_4) * std::exp(-4.0 * xi_a_sq) / (8.0 * std::sqrt(PI) * xi_a_5) +
                3.0 * std::erfc(2.0 * xi_a)
            );
            host_self_G2_uniq[i] = val_1 / std::pow(a_j, 5.0);
        }
        CUDA_CHECK(cudaMemcpy(d_self_G2_uniq, host_self_G2_uniq.data(), num_unique_radii * sizeof(double), cudaMemcpyHostToDevice));

        std::vector<double> host_field_quad_1(table_size, 0.0);
        std::vector<double> host_field_quad_2(table_size, 0.0);
        std::vector<double> host_field_quad_3(table_size, 0.0);
        std::vector<double> host_grad_quad_1(table_size, 0.0);
        std::vector<double> host_grad_quad_2(table_size, 0.0);
        std::vector<double> host_grad_quad_3(table_size, 0.0);
        std::vector<double> host_grad_quad_4(table_size, 0.0);

        double xi_7 = xi_6 * xi;
        double xi_8 = xi_7 * xi;
        double xi_9 = xi_8 * xi;
        double xi_10 = xi_9 * xi;

        double self_G2_val = 1.0 / (2.0 * PI) * (
            3.0 * (3.0 - 10.0 * xi_sq + 20.0 * xi_4) / (8.0 * std::sqrt(PI) * xi_5) -
            3.0 * (3.0 + 2.0 * xi_sq + 4.0 * xi_4) * std::exp(-4.0 * xi_sq) / (8.0 * std::sqrt(PI) * xi_5) +
            3.0 * std::erfc(2.0 * xi)
        );

        host_field_quad_1[0] = 0.0;
        host_field_quad_2[0] = 0.0;
        host_field_quad_3[0] = 0.0;
        host_grad_quad_1[0] = self_G2_val;
        host_grad_quad_2[0] = self_G2_val;
        host_grad_quad_3[0] = self_G2_val;
        host_grad_quad_4[0] = self_G2_val;

        for (size_t idx = 0; idx < num_r_steps; ++idx) {
            double r = r_vals[idx];
            double r_sq = r * r;
            double r_cub = r_sq * r;
            double r_4 = r_sq * r_sq;
            double r_5 = r_4 * r;
            double r_6 = r_4 * r_sq;
            double r_8 = r_4 * r_4;
            double r_10 = r_5 * r_5;

            // --- field_quad_1 calculation ---
            double fq1_exppolyp = 15.0 / (16384.0 * pi_pow_1_5 * xi_7 * r_4) * 
                (-24.0 * xi_6 * r_6 * r - 4.0 * xi_4 * (9.0 - 8.0 * xi_sq) * r_5 + 8.0 * xi_4 * (3.0 - 8.0 * xi_sq) * r_4 + 
                 48.0 * xi_6 * r_6 + 2.0 * xi_sq * (21.0 - 80.0 * xi_sq + 64.0 * xi_4) * r_cub - 
                 4.0 * xi_sq * std::pow(3.0 - 8.0 * xi_sq, 2.0) * r_sq - 
                 (45.0 - 120.0 * xi_sq + 192.0 * xi_4 - 512.0 * xi_6) * r - 
                 2.0 * (45.0 + 24.0 * xi_sq - 64.0 * xi_4 + 512.0 * xi_6));

            double fq1_exppolym = 15.0 / (16384.0 * pi_pow_1_5 * xi_7 * r_4) * 
                (-24.0 * xi_6 * r_6 * r - 4.0 * xi_4 * (9.0 - 8.0 * xi_sq) * r_5 - 8.0 * xi_4 * (3.0 - 8.0 * xi_sq) * r_4 - 
                 48.0 * xi_6 * r_6 + 2.0 * xi_sq * (21.0 - 80.0 * xi_sq + 64.0 * xi_4) * r_cub + 
                 4.0 * xi_sq * std::pow(3.0 - 8.0 * xi_sq, 2.0) * r_sq - 
                 (45.0 - 120.0 * xi_sq + 192.0 * xi_4 - 512.0 * xi_6) * r + 
                 2.0 * (45.0 + 24.0 * xi_sq - 64.0 * xi_4 + 512.0 * xi_6));

            double fq1_exppoly0 = 15.0 / (8192.0 * pi_pow_1_5 * xi_7 * r_cub) * 
                (24.0 * xi_6 * r_6 + 4.0 * xi_4 * (9.0 - 32.0 * xi_sq) * r_4 - 2.0 * xi_sq * (21.0 - 128.0 * xi_sq) * r_sq + 45.0 - 480.0 * xi_sq);

            double fq1_erfpolyp = 15.0 / (32768.0 * PI * xi_8 * r_4) * 
                (48.0 * xi_8 * r_6 * r_sq + 32.0 * xi_6 * (3.0 - 8.0 * xi_sq) * r_6 - 
                 24.0 * xi_4 * (3.0 - 16.0 * xi_sq) * r_4 + 72.0 * xi_sq * (1.0 - 8.0 * xi_sq) * r_sq - 45.0 + 480.0 * xi_sq + 4096.0 * xi_8);

            double fq1_erfpolym = 15.0 / (32768.0 * PI * xi_8 * r_4) * 
                (48.0 * xi_8 * r_6 * r_sq + 32.0 * xi_6 * (3.0 - 8.0 * xi_sq) * r_6 - 
                 24.0 * xi_4 * (3.0 - 16.0 * xi_sq) * r_4 + 72.0 * xi_sq * (1.0 - 8.0 * xi_sq) * r_sq - 45.0 + 480.0 * xi_sq + 4096.0 * xi_8);

            double fq1_erfpoly0 = 15.0 / (16384.0 * PI * xi_8 * r_4) * 
                (-48.0 * xi_8 * r_6 * r_sq - 32.0 * xi_6 * (3.0 - 8.0 * xi_sq) * r_6 + 
                 24.0 * xi_4 * (3.0 - 16.0 * xi_sq) * r_4 - 72.0 * xi_sq * (1.0 - 8.0 * xi_sq) * r_sq + 45.0 - 480.0 * xi_sq);

            double fq1_reg = -15.0 / (4.0 * PI * r_4) + 15.0 * r_sq / (64.0 * PI) * (1.0 - 3.0 * r_sq / 16.0);

            host_field_quad_1[idx + 1] = fq1_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + fq1_erfpolyp * std::erfc((r + 2.0) * xi) +
                                         fq1_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + fq1_erfpolym * std::erfc((r - 2.0) * xi) +
                                         fq1_exppoly0 * std::exp(-r_sq * xi_sq) + fq1_erfpoly0 * std::erfc(r * xi) +
                                         ((r < 2.0) ? fq1_reg : 0.0);

            // --- field_quad_2 calculation ---
            double fq2_exppolyp = 3.0 / (16384.0 * pi_pow_1_5 * xi_7 * r_4) * 
                (-40.0 * xi_6 * r_6 * r + 80.0 * xi_6 * r_6 - 20.0 * xi_4 * (11.0 - 24.0 * xi_sq) * r_5 + 
                 8.0 * xi_4 * (45.0 + 8.0 * xi_sq) * r_4 - 2.0 * xi_sq * (45.0 - 80.0 * xi_sq + 64.0 * xi_4) * r_cub - 
                 4.0 * xi_sq * (15.0 + 48.0 * xi_sq - 64.0 * xi_4) * r_sq + 
                 (45.0 - 120.0 * xi_sq + 192.0 * xi_4 - 512.0 * xi_6) * r + 
                 2.0 * (45.0 + 24.0 * xi_sq - 64.0 * xi_4 + 512.0 * xi_6));

            double fq2_exppolym = 3.0 / (16384.0 * pi_pow_1_5 * xi_7 * r_4) * 
                (-40.0 * xi_6 * r_6 * r - 80.0 * xi_6 * r_6 - 20.0 * xi_4 * (11.0 - 24.0 * xi_sq) * r_5 - 
                 8.0 * xi_4 * (45.0 + 8.0 * xi_sq) * r_4 - 2.0 * xi_sq * (45.0 - 80.0 * xi_sq + 64.0 * xi_4) * r_cub + 
                 4.0 * xi_sq * (15.0 + 48.0 * xi_sq - 64.0 * xi_4) * r_sq + 
                 (45.0 - 120.0 * xi_sq + 192.0 * xi_4 - 512.0 * xi_6) * r - 
                 2.0 * (45.0 + 24.0 * xi_sq - 64.0 * xi_4 + 512.0 * xi_6));

            double fq2_exppoly0 = 15.0 / (8192.0 * pi_pow_1_5 * xi_7 * r_cub) * 
                (8.0 * xi_6 * r_6 + 4.0 * xi_4 * (11.0 - 32.0 * xi_sq) * r_4 + 2.0 * xi_sq * (9.0 - 64.0 * xi_sq) * r_sq - 9.0 + 96.0 * xi_sq);

            double fq2_erfpolyp = 3.0 / (32768.0 * PI * xi_8 * r_4) * 
                (80.0 * xi_8 * r_6 * r_sq + 160.0 * xi_6 * (3.0 - 8.0 * xi_sq) * r_6 - 2048.0 * xi_8 * r_5 + 
                 120.0 * xi_4 * (3.0 - 16.0 * xi_sq) * r_4 - 120.0 * xi_sq * (1.0 - 8.0 * xi_sq) * r_sq + 
                 45.0 - 480.0 * xi_sq - 4096.0 * xi_8);

            double fq2_erfpolym = 3.0 / (32768.0 * PI * xi_8 * r_4) * 
                (80.0 * xi_8 * r_6 * r_sq + 160.0 * xi_6 * (3.0 - 8.0 * xi_sq) * r_6 + 2048.0 * xi_8 * r_5 + 
                 120.0 * xi_4 * (3.0 - 16.0 * xi_sq) * r_4 - 120.0 * xi_sq * (1.0 - 8.0 * xi_sq) * r_sq + 
                 45.0 - 480.0 * xi_sq - 4096.0 * xi_8);

            double fq2_erfpoly0 = 15.0 / (16384.0 * PI * xi_8 * r_4) * 
                (-16.0 * xi_8 * r_6 * r_sq - 32.0 * xi_6 * (3.0 - 8.0 * xi_sq) * r_6 - 24.0 * xi_4 * (3.0 - 16.0 * xi_sq) * r_4 + 
                 24.0 * xi_sq * (1.0 - 8.0 * xi_sq) * r_sq - 9.0 + 96.0 * xi_sq);

            double fq2_reg = 3.0 / (4.0 * PI * r_4) - 3.0 * r / (8.0 * PI) * (1.0 - 5.0 * r / 8.0 + 5.0 * r_cub / 128.0);

            host_field_quad_2[idx + 1] = fq2_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + fq2_erfpolyp * std::erfc((r + 2.0) * xi) +
                                         fq2_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + fq2_erfpolym * std::erfc((r - 2.0) * xi) +
                                         fq2_exppoly0 * std::exp(-r_sq * xi_sq) + fq2_erfpoly0 * std::erfc(r * xi) +
                                         ((r < 2.0) ? fq2_reg : 0.0);

            // --- field_quad_3 calculation ---
            double fq3_exppolyp = 5.0 / (1024.0 * pi_pow_1_5 * xi_5 * r_sq) * 
                (4.0 * xi_4 * r_5 - 8.0 * xi_4 * r_4 + 16.0 * xi_sq * (1.0 - 2.0 * xi_sq) * r_cub - 24.0 * xi_sq * r_sq + 3.0 * r + 6.0);

            double fq3_exppolym = 5.0 / (1024.0 * pi_pow_1_5 * xi_5 * r_sq) * 
                (4.0 * xi_4 * r_5 + 8.0 * xi_4 * r_4 + 16.0 * xi_sq * (1.0 - 2.0 * xi_sq) * r_cub + 24.0 * xi_sq * r_sq + 3.0 * r - 6.0);

            double fq3_exppoly0 = 5.0 / (512.0 * pi_pow_1_5 * xi_5 * r) * 
                (-4.0 * xi_4 * r_4 - 16.0 * xi_sq * (1.0 - 3.0 * xi_sq) * r_sq - 3.0 + 24.0 * xi_sq);

            double fq3_erfpolyp = 5.0 / (2048.0 * PI * xi_6 * r_sq) * 
                (-8.0 * xi_6 * r_6 - 12.0 * xi_4 * (3.0 - 8.0 * xi_sq) * r_4 + 128.0 * xi_6 * r_cub - 6.0 * xi_sq * (3.0 - 16.0 * xi_sq) * r_sq + 3.0 - 24.0 * xi_sq);

            double fq3_erfpolym = 5.0 / (2048.0 * PI * xi_6 * r_sq) * 
                (-8.0 * xi_6 * r_6 - 12.0 * xi_4 * (3.0 - 8.0 * xi_sq) * r_4 - 128.0 * xi_6 * r_cub - 6.0 * xi_sq * (3.0 - 16.0 * xi_sq) * r_sq + 3.0 - 24.0 * xi_sq);

            double fq3_erfpoly0 = 5.0 / (1024.0 * PI * xi_6 * r_sq) * 
                (8.0 * xi_6 * r_6 + 12.0 * xi_4 * (3.0 - 8.0 * xi_sq) * r_4 + 6.0 * xi_sq * (3.0 - 16.0 * xi_sq) * r_sq - 3.0 + 24.0 * xi_sq);

            double fq3_reg = 5.0 * r / (8.0 * PI) * (1.0 - 3.0 * r / 4.0 + r_cub / 16.0);

            host_field_quad_3[idx + 1] = fq3_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + fq3_erfpolyp * std::erfc((r + 2.0) * xi) +
                                         fq3_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + fq3_erfpolym * std::erfc((r - 2.0) * xi) +
                                         fq3_exppoly0 * std::exp(-r_sq * xi_sq) + fq3_erfpoly0 * std::erfc(r * xi) +
                                         ((r < 2.0) ? fq3_reg : 0.0);

            // --- grad_quad_1 calculation ---
            double gq1_exppolyp = 75.0 / (16384.0 * pi_pow_1_5 * xi_7 * r_cub) * 
                (8.0 * xi_6 * r_6 * r - 16.0 * xi_6 * r_6 + (44.0 * xi_4 - 32.0 * xi_6) * r_5 - (72.0 * xi_4 - 64.0 * xi_6) * r_4 + 
                 (18.0 * xi_sq + 32.0 * xi_4) * r_cub + (12.0 * xi_sq - 64.0 * xi_4) * r_sq - (9.0 + 24.0 * xi_sq) * r - (18.0 - 48.0 * xi_sq));

            double gq1_exppolym = 75.0 / (16384.0 * pi_pow_1_5 * xi_7 * r_cub) * 
                (8.0 * xi_6 * r_6 * r + 16.0 * xi_6 * r_6 + (44.0 * xi_4 - 32.0 * xi_6) * r_5 + (72.0 * xi_4 - 64.0 * xi_6) * r_4 + 
                 (18.0 * xi_sq + 32.0 * xi_4) * r_cub - (12.0 * xi_sq - 64.0 * xi_4) * r_sq - (9.0 + 24.0 * xi_sq) * r + (18.0 - 48.0 * xi_sq));

            double gq1_exppoly0 = -75.0 / (8192.0 * pi_pow_1_5 * xi_7 * r_sq) * 
                (8.0 * xi_6 * r_6 + (44.0 * xi_4 - 64.0 * xi_6) * r_4 + (18.0 * xi_sq - 64.0 * xi_4 + 128.0 * xi_6) * r_sq - 9.0 + 48.0 * xi_sq - 192.0 * xi_4);

            double gq1_erfpolyp = -75.0 / (32768.0 * PI * xi_8 * r_cub) * 
                (16.0 * xi_8 * r_6 * r_sq + (96.0 * xi_6 - 128.0 * xi_8) * r_6 + (72.0 * xi_4 - 192.0 * xi_6 + 256.0 * xi_8) * r_4 - 
                 (24.0 * xi_sq - 96.0 * xi_4 + 256.0 * xi_6) * r_sq + 9.0 - 48.0 * xi_sq + 192.0 * xi_4);

            double gq1_erfpolym = -75.0 / (32768.0 * PI * xi_8 * r_cub) * 
                (16.0 * xi_8 * r_6 * r_sq + (96.0 * xi_6 - 128.0 * xi_8) * r_6 + (72.0 * xi_4 - 192.0 * xi_6 + 256.0 * xi_8) * r_4 - 
                 (24.0 * xi_sq - 96.0 * xi_4 + 256.0 * xi_6) * r_sq + 9.0 - 48.0 * xi_sq + 192.0 * xi_4);

            double gq1_erfpoly0 = 75.0 / (16384.0 * PI * xi_8 * r_cub) * 
                (16.0 * xi_8 * r_6 * r_sq + (96.0 * xi_6 - 128.0 * xi_8) * r_6 + (72.0 * xi_4 - 192.0 * xi_6 + 256.0 * xi_8) * r_4 - 
                 (24.0 * xi_sq - 96.0 * xi_4 + 256.0 * xi_6) * r_sq + 9.0 - 48.0 * xi_sq + 192.0 * xi_4);

            double gq1_reg = 75.0 * r * std::pow(4.0 - r_sq, 2.0) / (1024.0 * PI);

            host_grad_quad_1[idx + 1] = gq1_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + gq1_erfpolyp * std::erfc((r + 2.0) * xi) +
                                         gq1_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + gq1_erfpolym * std::erfc((r - 2.0) * xi) +
                                         gq1_exppoly0 * std::exp(-r_sq * xi_sq) + gq1_erfpoly0 * std::erfc(r * xi) +
                                         ((r < 2.0) ? gq1_reg : 0.0);

            // --- grad_quad_2 calculation ---
            double gq2_exppolyp = -15.0 / (16384.0 * pi_pow_1_5 * xi_7 * r_cub) * 
                (24.0 * xi_6 * r_6 * r - 48.0 * xi_6 * r_6 + (148.0 * xi_4 - 96.0 * xi_6) * r_5 - (264.0 * xi_4 - 192.0 * xi_6) * r_4 + 
                 (90.0 * xi_sq + 72.0 * xi_4 + 256.0 * xi_8) * r_cub + (60.0 * xi_sq - 144.0 * xi_4 + 512.0 * xi_8) * r_sq - 
                 (45.0 + 120.0 * xi_sq - 192.0 * xi_4 + 512.0 * xi_6) * r - 2.0 * (45.0 + 24.0 * xi_sq - 64.0 * xi_4 + 512.0 * xi_6));

            double gq2_exppolym = -15.0 / (16384.0 * pi_pow_1_5 * xi_7 * r_cub) * 
                (24.0 * xi_6 * r_6 * r + 48.0 * xi_6 * r_6 + (148.0 * xi_4 - 96.0 * xi_6) * r_5 + (264.0 * xi_4 - 192.0 * xi_6) * r_4 + 
                 (90.0 * xi_sq + 72.0 * xi_4 + 256.0 * xi_8) * r_cub - (60.0 * xi_sq - 144.0 * xi_4 + 512.0 * xi_8) * r_sq - 
                 (45.0 + 120.0 * xi_sq - 192.0 * xi_4 + 512.0 * xi_6) * r + 2.0 * (45.0 + 24.0 * xi_sq - 64.0 * xi_4 + 512.0 * xi_6));

            double gq2_exppoly0 = 75.0 / (8192.0 * pi_pow_1_5 * xi_7 * r_sq) * 
                (8.0 * xi_6 * r_6 + (44.0 * xi_4 - 64.0 * xi_6) * r_4 + (18.0 * xi_sq - 48.0 * xi_4 + 128.0 * xi_6) * r_sq - 9.0 + 40.0 * xi_sq - 192.0 * xi_4);

            double gq2_erfpolyp = -15.0 / (32768.0 * PI * xi_8 * r_cub) * 
                (48.0 * xi_8 * r_6 * r_sq + (288.0 * xi_6 - 384.0 * xi_8) * r_6 - 4096.0 * xi_8 * r_5 + 
                 (216.0 * xi_4 - 576.0 * xi_6 + 768.0 * xi_8) * r_4 - (72.0 * xi_sq - 288.0 * xi_4 + 768.0 * xi_6) * r_sq + 
                 45.0 - 240.0 * xi_sq + 960.0 * xi_4 - 8192.0 * xi_8);

            double gq2_erfpolym = -15.0 / (32768.0 * PI * xi_8 * r_cub) * 
                (48.0 * xi_8 * r_6 * r_sq + (288.0 * xi_6 - 384.0 * xi_8) * r_6 + 4096.0 * xi_8 * r_5 + 
                 (216.0 * xi_4 - 576.0 * xi_6 + 768.0 * xi_8) * r_4 - (72.0 * xi_sq - 288.0 * xi_4 + 768.0 * xi_6) * r_sq + 
                 45.0 - 240.0 * xi_sq + 960.0 * xi_4 - 8192.0 * xi_8);

            double gq2_erfpoly0 = 15.0 / (16384.0 * PI * xi_8 * r_cub) * 
                (48.0 * xi_8 * r_6 * r_sq + (288.0 * xi_6 - 384.0 * xi_8) * r_6 + (216.0 * xi_4 - 576.0 * xi_6 + 768.0 * xi_8) * r_4 - 
                 (72.0 * xi_sq - 288.0 * xi_4 + 768.0 * xi_6) * r_sq + 45.0 - 240.0 * xi_sq + 960.0 * xi_4);

            double gq2_reg = -45.0 * r / (32.0 * PI) * (1.0 - 5.0 * r_sq / 12.0 + r_4 / 16.0);

            host_grad_quad_2[idx + 1] = gq2_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + gq2_erfpolyp * std::erfc((r + 2.0) * xi) +
                                         gq2_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + gq2_erfpolym * std::erfc((r - 2.0) * xi) +
                                         gq2_exppoly0 * std::exp(-r_sq * xi_sq) + gq2_erfpoly0 * std::erfc(r * xi) +
                                         ((r < 2.0) ? gq2_reg : 0.0);

            // --- grad_quad_3 calculation ---
            double gq3_exppolyp = -15.0 / (65536.0 * pi_pow_1_5 * xi_9 * r_5) * 
                (48.0 * xi_8 * r_6 * r_cub - 96.0 * xi_8 * r_6 * r_sq + (336.0 * xi_6 - 288.0 * xi_8) * r_6 * r - 576.0 * (xi_6 - xi_8) * r_6 - 
                 (216.0 * xi_4 + 144.0 * xi_6 + 128.0 * xi_8) * r_5 + (480.0 * xi_6 + 256.0 * xi_8) * r_4 - 
                 (180.0 * xi_sq - 120.0 * xi_4 + 640.0 * xi_6 - 512.0 * xi_8) * r_cub + 
                 (720.0 * xi_4 + 768.0 * xi_6 - 1024.0 * xi_8) * r_sq + 
                 (135.0 + 180.0 * xi_sq + 480.0 * xi_4 - 768.0 * xi_6 + 2048.0 * xi_8) * r - 
                 (-270.0 + 1080.0 * xi_sq + 192.0 * xi_4 - 512.0 * xi_6 + 4096.0 * xi_8));

            double gq3_exppolym = -15.0 / (65536.0 * pi_pow_1_5 * xi_9 * r_5) * 
                (48.0 * xi_8 * r_6 * r_cub + 96.0 * xi_8 * r_6 * r_sq + (336.0 * xi_6 - 288.0 * xi_8) * r_6 * r + 576.0 * (xi_6 - xi_8) * r_6 + 
                 (216.0 * xi_4 + 144.0 * xi_6 + 128.0 * xi_8) * r_5 + (480.0 * xi_6 + 256.0 * xi_8) * r_4 - 
                 (180.0 * xi_sq - 120.0 * xi_4 + 640.0 * xi_6 - 512.0 * xi_8) * r_cub - 
                 (720.0 * xi_4 + 768.0 * xi_6 - 1024.0 * xi_8) * r_sq + 
                 (135.0 + 180.0 * xi_sq + 480.0 * xi_4 - 768.0 * xi_6 + 2048.0 * xi_8) * r + 
                 (-270.0 + 1080.0 * xi_sq + 192.0 * xi_4 - 512.0 * xi_6 + 4096.0 * xi_8));

            double gq3_exppoly0 = 15.0 / (32768.0 * pi_pow_1_5 * xi_9 * r_4) * 
                (48.0 * xi_8 * r_8 + (336.0 * xi_6 - 480.0 * xi_8) * r_6 + (216.0 * xi_4 - 720.0 * xi_6 + 1280.0 * xi_8) * r_4 - 
                 (180.0 * xi_sq - 840.0 * xi_4 + 2560.0 * xi_6) * r_sq + 135.0 - 900.0 * xi_sq + 4800.0 * xi_4);

            double gq3_erfpolyp = -15.0 / (131072.0 * PI * xi_10 * r_5) * 
                (-96.0 * xi_10 * r_10 + (-720.0 * xi_8 + 960.0 * xi_10) * r_8 - (720.0 * xi_6 - 1920.0 * xi_8 + 2560.0 * xi_10) * r_6 + 
                 (360.0 * xi_4 - 1440.0 * xi_6 + 3840.0 * xi_8) * r_4 - (270.0 * xi_sq - 1440.0 * xi_4 + 5760.0 * xi_6) * r_sq + 
                 135.0 - 900.0 * xi_sq + 4800.0 * xi_4 + 16384.0 * xi_10);

            double gq3_erfpolym = -15.0 / (131072.0 * PI * xi_10 * r_5) * 
                (-96.0 * xi_10 * r_10 + (-720.0 * xi_8 + 960.0 * xi_10) * r_8 - (720.0 * xi_6 - 1920.0 * xi_8 + 2560.0 * xi_10) * r_6 + 
                 (360.0 * xi_4 - 1440.0 * xi_6 + 3840.0 * xi_8) * r_4 - (270.0 * xi_sq - 1440.0 * xi_4 + 5760.0 * xi_6) * r_sq + 
                 135.0 - 900.0 * xi_sq + 4800.0 * xi_4 + 16384.0 * xi_10);

            double gq3_erfpoly0 = -15.0 / (65536.0 * PI * xi_10 * r_5) * 
                (96.0 * xi_10 * r_10 + (720.0 * xi_8 - 960.0 * xi_10) * r_8 + (720.0 * xi_6 - 1920.0 * xi_8 + 2560.0 * xi_10) * r_6 - 
                 (360.0 * xi_4 - 1440.0 * xi_6 + 3840.0 * xi_8) * r_4 + (270.0 * xi_sq - 1440.0 * xi_4 + 5760.0 * xi_6) * r_sq - 
                 135.0 + 900.0 * xi_sq - 4800.0 * xi_4);

            double gq3_reg = -15.0 * std::pow(r_sq - 4.0, 3) * (3.0 * r_4 + 6.0 * r_sq + 8.0) / (2048.0 * PI * r_5);

            host_grad_quad_3[idx + 1] = gq3_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + gq3_erfpolyp * std::erfc((r + 2.0) * xi) +
                                         gq3_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + gq3_erfpolym * std::erfc((r - 2.0) * xi) +
                                         gq3_exppoly0 * std::exp(-r_sq * xi_sq) + gq3_erfpoly0 * std::erfc(r * xi) +
                                         ((r < 2.0) ? gq3_reg : 0.0);

            // --- grad_quad_4 calculation ---
            double gq4_exppolyp = -15.0 / (65536.0 * pi_pow_1_5 * xi_9 * r_5) * 
                (144.0 * xi_8 * r_6 * r_cub - 288.0 * xi_8 * r_6 * r_sq + (288.0 * xi_6 + 96.0 * xi_8) * r_6 * r - (288.0 * xi_6 + 192.0 * xi_8) * r_6 - 
                 (432.0 * xi_4 - 912.0 * xi_6 + 896.0 * xi_8) * r_5 + (720.0 * xi_4 - 480.0 * xi_6 + 1792.0 * xi_8) * r_4 + 
                 (720.0 * xi_sq - 2280.0 * xi_4 + 4480.0 * xi_6 - 3584.0 * xi_8) * r_cub - 
                 (1080.0 * xi_sq + 2160.0 * xi_4 + 5376.0 * xi_6 - 7168.0 * xi_8) * r_sq - 
                 (945.0 + 1260.0 * xi_sq + 3360.0 * xi_4 - 5376.0 * xi_6 + 14336.0 * xi_8) * r - 
                 (1890.0 - 7560.0 * xi_sq - 1344.0 * xi_4 + 3584.0 * xi_6 - 28672.0 * xi_8));

            double gq4_exppolym = -15.0 / (65536.0 * pi_pow_1_5 * xi_9 * r_5) * 
                (144.0 * xi_8 * r_6 * r_cub + 288.0 * xi_8 * r_6 * r_sq + (288.0 * xi_6 + 96.0 * xi_8) * r_6 * r + (288.0 * xi_6 + 192.0 * xi_8) * r_6 - 
                 (432.0 * xi_4 - 912.0 * xi_6 + 896.0 * xi_8) * r_5 - (720.0 * xi_4 - 480.0 * xi_6 + 1792.0 * xi_8) * r_4 + 
                 (720.0 * xi_sq - 2280.0 * xi_4 + 4480.0 * xi_6 - 3584.0 * xi_8) * r_cub + 
                 (1080.0 * xi_sq + 2160.0 * xi_4 + 5376.0 * xi_6 - 7168.0 * xi_8) * r_sq - 
                 (945.0 + 1260.0 * xi_sq + 3360.0 * xi_4 - 5376.0 * xi_6 + 14336.0 * xi_8) * r + 
                 (1890.0 - 7560.0 * xi_sq - 1344.0 * xi_4 + 3584.0 * xi_6 - 28672.0 * xi_8));

            double gq4_exppoly0 = 15.0 / (32768.0 * pi_pow_1_5 * xi_9 * r_4) * 
                (144.0 * xi_8 * r_8 + (288.0 * xi_6 - 480.0 * xi_8) * r_6 - (432.0 * xi_4 - 1200.0 * xi_6 + 1280.0 * xi_8) * r_4 + 
                 (720.0 * xi_sq - 3000.0 * xi_4 + 6400.0 * xi_6) * r_sq - 945.0 + 6300.0 * xi_sq - 33600.0 * xi_4);

            double gq4_erfpolyp = 15.0 / (131072.0 * PI * xi_10 * r_5) * 
                (288.0 * xi_10 * r_10 + (720.0 * xi_8 - 960.0 * xi_10) * r_8 - (720.0 * xi_6 - 1920.0 * xi_8 + 2560.0 * xi_10) * r_6 + 
                 (1080.0 * xi_4 - 4320.0 * xi_6 + 11520.0 * xi_8) * r_4 - (1350.0 * xi_sq - 7200.0 * xi_4 + 28800.0 * xi_6) * r_sq + 
                 945.0 - 6300.0 * xi_sq + 33600.0 * xi_4 + 114688.0 * xi_10);

            double gq4_erfpolym = 15.0 / (131072.0 * PI * xi_10 * r_5) * 
                (288.0 * xi_10 * r_10 + (720.0 * xi_8 - 960.0 * xi_10) * r_8 - (720.0 * xi_6 - 1920.0 * xi_8 + 2560.0 * xi_10) * r_6 + 
                 (1080.0 * xi_4 - 4320.0 * xi_6 + 11520.0 * xi_8) * r_4 - (1350.0 * xi_sq - 7200.0 * xi_4 + 28800.0 * xi_6) * r_sq + 
                 945.0 - 6300.0 * xi_sq + 33600.0 * xi_4 + 114688.0 * xi_10);

            double gq4_erfpoly0 = -15.0 / (65536.0 * PI * xi_10 * r_5) * 
                (288.0 * xi_10 * r_10 + (720.0 * xi_8 - 960.0 * xi_10) * r_8 - (720.0 * xi_6 - 1920.0 * xi_8 + 2560.0 * xi_10) * r_6 + 
                 (1080.0 * xi_4 - 4320.0 * xi_6 + 11520.0 * xi_8) * r_4 - (1350.0 * xi_sq - 7200.0 * xi_4 + 28800.0 * xi_6) * r_sq + 
                 945.0 - 6300.0 * xi_sq + 33600.0 * xi_4);

            double gq4_reg = -15.0 * (3584.0 - 80.0 * r_6 - 30.0 * std::pow(r, 8.0) + 9.0 * std::pow(r, 10.0)) / (2048.0 * PI * r_5);

            host_grad_quad_4[idx + 1] = gq4_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + gq4_erfpolyp * std::erfc((r + 2.0) * xi) +
                                         gq4_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + gq4_erfpolym * std::erfc((r - 2.0) * xi) +
                                         gq4_exppoly0 * std::exp(-r_sq * xi_sq) + gq4_erfpoly0 * std::erfc(r * xi) +
                                         ((r < 2.0) ? gq4_reg : 0.0);
        }

        if (d_field_quad_1) CUDA_CHECK(cudaFree(d_field_quad_1));
        if (d_field_quad_2) CUDA_CHECK(cudaFree(d_field_quad_2));
        if (d_field_quad_3) CUDA_CHECK(cudaFree(d_field_quad_3));
        if (d_grad_quad_1) CUDA_CHECK(cudaFree(d_grad_quad_1));
        if (d_grad_quad_2) CUDA_CHECK(cudaFree(d_grad_quad_2));
        if (d_grad_quad_3) CUDA_CHECK(cudaFree(d_grad_quad_3));
        if (d_grad_quad_4) CUDA_CHECK(cudaFree(d_grad_quad_4));

        CUDA_CHECK(cudaMalloc(&d_field_quad_1, table_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_field_quad_2, table_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_field_quad_3, table_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_quad_1, table_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_quad_2, table_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_quad_3, table_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_quad_4, table_size * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_field_quad_1, host_field_quad_1.data(), table_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_field_quad_2, host_field_quad_2.data(), table_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_field_quad_3, host_field_quad_3.data(), table_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_quad_1, host_grad_quad_1.data(), table_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_quad_2, host_grad_quad_2.data(), table_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_quad_3, host_grad_quad_3.data(), table_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_quad_4, host_grad_quad_4.data(), table_size * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void Polydisperse_Ewald_Electric_Field::computePrecalculations() {
    rc = std::sqrt(-std::log(errortol)) / xi;

    if (rc > box_x / 2.0 || rc > box_y / 2.0 || rc > box_z / 2.0) {
        throw std::runtime_error("Real space cutoff (" + std::to_string(rc) + ") larger than half the box length.");
    }

    double kcut = 2.0 * xi * xi * rc;
    const double PI = 3.14159265358979323846;

    num_grid[0] = static_cast<int>(std::ceil(1.0 + box_x * kcut / PI));
    num_grid[1] = static_cast<int>(std::ceil(1.0 + box_y * kcut / PI));
    num_grid[2] = static_cast<int>(std::ceil(1.0 + box_z * kcut / PI));

    grid_spacing[0] = box_x / num_grid[0];
    grid_spacing[1] = box_y / num_grid[1];
    grid_spacing[2] = box_z / num_grid[2];

    double mean_h = (grid_spacing[0] + grid_spacing[1] + grid_spacing[2]) / 3.0;

    double P_val = std::ceil(2.0 / mean_h - 2.0 * std::log(errortol) / PI);
    P_support = static_cast<int>(P_val);

    eta_scalar = P_support * std::pow(mean_h * xi, 2.0) / PI;

    int off_min = -static_cast<int>(std::floor((P_support - 1) / 2.0));
    int off_max = P_support + off_min;

    std::vector<int> host_offset;
    std::vector<double> host_offsetxyz;

    for (int x_off = off_min; x_off < off_max; ++x_off) {
        for (int y_off = off_min; y_off < off_max; ++y_off) {
            for (int z_off = off_min; z_off < off_max; ++z_off) {
                host_offset.push_back(x_off);
                host_offset.push_back(y_off);
                host_offset.push_back(z_off);

                host_offsetxyz.push_back(x_off * grid_spacing[0]);
                host_offsetxyz.push_back(y_off * grid_spacing[1]);
                host_offsetxyz.push_back(z_off * grid_spacing[2]);
            }
        }
    }

    num_offsets = host_offset.size() / 3;

    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    std::vector<double> host_scale_coef(grid_voxels, 0.0);
    std::vector<double> host_scale_coef_Q_imag(grid_voxels, 0.0);
    std::vector<double> host_scale_coef_GP_imag(grid_voxels, 0.0);
    std::vector<double> host_scale_coef_GQ_real(grid_voxels, 0.0);
    std::vector<double> host_Qfactor(grid_voxels * 5, 0.0);
    std::vector<double> host_Qfactor_dot(grid_voxels * 5, 0.0);

    auto getKvals = [](int N, double box_len) {
        std::vector<double> K(N);
        double start = -std::ceil((N - 1) / 2.0);
        for (int i = 0; i < N; ++i) {
            K[i] = (start + i) * 2.0 * 3.14159265358979323846 / box_len;
        }
        return K;
    };

    std::vector<double> Kx = getKvals(num_grid[0], box_x);
    std::vector<double> Ky = getKvals(num_grid[1], box_y);
    std::vector<double> Kz = getKvals(num_grid[2], box_z);

    int k0x = static_cast<int>(std::ceil((num_grid[0] - 1) / 2.0));
    int k0y = static_cast<int>(std::ceil((num_grid[1] - 1) / 2.0));
    int k0z = static_cast<int>(std::ceil((num_grid[2] - 1) / 2.0));

    for (int ix = 0; ix < num_grid[0]; ++ix) {
        for (int iy = 0; iy < num_grid[1]; ++iy) {
            for (int iz = 0; iz < num_grid[2]; ++iz) {
                size_t linear_idx = ix * num_grid[1] * num_grid[2] + iy * num_grid[2] + iz;

                if (ix == k0x && iy == k0y && iz == k0z) {
                    host_scale_coef[linear_idx] = 0.0;
                    if (solve_quadrupoles) {
                        host_scale_coef_Q_imag[linear_idx] = 0.0;
                        host_scale_coef_GP_imag[linear_idx] = 0.0;
                        host_scale_coef_GQ_real[linear_idx] = 0.0;
                        for (int c = 0; c < 5; ++c) {
                            host_Qfactor[linear_idx * 5 + c] = 0.0;
                            host_Qfactor_dot[linear_idx * 5 + c] = 0.0;
                        }
                    }
                    continue;
                }

                double kx_val = Kx[ix];
                double ky_val = Ky[iy];
                double kz_val = Kz[iz];

                double ksqsm = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;

                double exp_part = std::exp(-(1.0 - eta_scalar) * ksqsm / (4.0 * xi * xi));
                host_scale_coef[linear_idx] = exp_part / ksqsm;

                if (solve_quadrupoles) {
                    double kmag = std::sqrt(ksqsm);
                    double kh0 = kx_val / kmag;
                    double kh1 = ky_val / kmag;
                    double kh2 = kz_val / kmag;

                    host_Qfactor[linear_idx * 5 + 0] = kh0 * kh0 - 1.0 / 3.0;
                    host_Qfactor[linear_idx * 5 + 1] = kh0 * kh1;
                    host_Qfactor[linear_idx * 5 + 2] = kh0 * kh2;
                    host_Qfactor[linear_idx * 5 + 3] = kh1 * kh1 - 1.0 / 3.0;
                    host_Qfactor[linear_idx * 5 + 4] = kh1 * kh2;

                    host_Qfactor_dot[linear_idx * 5 + 0] = kh0 * kh0 - kh2 * kh2;
                    host_Qfactor_dot[linear_idx * 5 + 1] = 2.0 * kh0 * kh1;
                    host_Qfactor_dot[linear_idx * 5 + 2] = 2.0 * kh0 * kh2;
                    host_Qfactor_dot[linear_idx * 5 + 3] = kh1 * kh1 - kh2 * kh2;
                    host_Qfactor_dot[linear_idx * 5 + 4] = 2.0 * kh1 * kh2;

                    // Analytical Fourier transform of Gaussian spread functions for scalar potential/fields
                    host_scale_coef_Q_imag[linear_idx] = -0.5 * exp_part;
                    host_scale_coef_GP_imag[linear_idx] = exp_part;
                    host_scale_coef_GQ_real[linear_idx] = -0.5 * exp_part * ksqsm;
                }
            }
        }
    }

    if (d_offset) CUDA_CHECK(cudaFree(d_offset));
    if (d_offsetxyz) CUDA_CHECK(cudaFree(d_offsetxyz));
    if (d_scale_coef) CUDA_CHECK(cudaFree(d_scale_coef));
    if (d_scale_coef_Q_imag) CUDA_CHECK(cudaFree(d_scale_coef_Q_imag));
    if (d_scale_coef_GP_imag) CUDA_CHECK(cudaFree(d_scale_coef_GP_imag));
    if (d_scale_coef_GQ_real) CUDA_CHECK(cudaFree(d_scale_coef_GQ_real));
    if (d_Qfactor) CUDA_CHECK(cudaFree(d_Qfactor));
    if (d_Qfactor_dot) CUDA_CHECK(cudaFree(d_Qfactor_dot));

    CUDA_CHECK(cudaMalloc(&d_offset, num_offsets * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsetxyz, num_offsets * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scale_coef, grid_voxels * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_offset, host_offset.data(), num_offsets * 3 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsetxyz, host_offsetxyz.data(), num_offsets * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_coef, host_scale_coef.data(), grid_voxels * sizeof(double), cudaMemcpyHostToDevice));

    if (solve_quadrupoles) {
        CUDA_CHECK(cudaMalloc(&d_scale_coef_Q_imag, grid_voxels * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_scale_coef_GP_imag, grid_voxels * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_scale_coef_GQ_real, grid_voxels * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Qfactor, grid_voxels * 5 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Qfactor_dot, grid_voxels * 5 * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_scale_coef_Q_imag, host_scale_coef_Q_imag.data(), grid_voxels * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scale_coef_GP_imag, host_scale_coef_GP_imag.data(), grid_voxels * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scale_coef_GQ_real, host_scale_coef_GQ_real.data(), grid_voxels * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Qfactor, host_Qfactor.data(), grid_voxels * 5 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Qfactor_dot, host_Qfactor_dot.data(), grid_voxels * 5 * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        d_scale_coef_Q_imag = nullptr;
        d_scale_coef_GP_imag = nullptr;
        d_scale_coef_GQ_real = nullptr;
        d_Qfactor = nullptr;
        d_Qfactor_dot = nullptr;
    }
}













void Polydisperse_Ewald_Electric_Field::spreadPrecalcs() {
    if (num_particles == 0) return;

    num_spread = num_particles * num_offsets;
    size_t size_coef_bytes = num_spread * 3 * sizeof(double);
    size_t size_idxs_bytes = num_spread * sizeof(int);

    if (d_spread_coef) CUDA_CHECK(cudaFree(d_spread_coef));
    if (d_spread_idxs) CUDA_CHECK(cudaFree(d_spread_idxs));
    if (d_spread_coef_Q) { CUDA_CHECK(cudaFree(d_spread_coef_Q)); d_spread_coef_Q = nullptr; }

    CUDA_CHECK(cudaMalloc(&d_spread_coef, size_coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_spread_idxs, size_idxs_bytes));
    if (solve_quadrupoles) {
        CUDA_CHECK(cudaMalloc(&d_spread_coef_Q, num_spread * sizeof(double)));
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_spread + threadsPerBlock - 1) / threadsPerBlock;

    spread_precalcs_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
        d_x_part, d_y_part, d_z_part,
        d_radii,
        d_offset, d_offsetxyz,
        d_spread_coef, d_spread_idxs,
        num_particles, num_offsets,
        grid_spacing[0], grid_spacing[1], grid_spacing[2],
        num_grid[0], num_grid[1], num_grid[2],
        eta_scalar, xi
    );
    CUDA_CHECK(cudaGetLastError());

    if (solve_quadrupoles) {
        double const_factor = std::pow(2.0 * xi * xi / 3.14159265358979323846, 1.5) * grid_spacing[0] * grid_spacing[1] * grid_spacing[2];
        spread_precalcs_kernel_point<<<blocksPerGrid, threadsPerBlock>>>(
            d_x_part, d_y_part, d_z_part,
            d_offset, d_offsetxyz,
            d_spread_coef_Q,
            num_particles, num_offsets,
            grid_spacing[0], grid_spacing[1], grid_spacing[2],
            eta_scalar, xi,
            const_factor
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Polydisperse_Ewald_Electric_Field::contractPrecalcs() {
    if (num_field_points == 0) return;

    num_contract = num_field_points * num_offsets;
    size_t size_coef_bytes = num_contract * 3 * sizeof(double);
    size_t size_idxs_bytes = num_contract * sizeof(int);
    size_t size_idx_bytes = num_contract * sizeof(int);
    size_t size_epoint_bytes = (num_field_points * 3 + num_quads * 5) * 2 * sizeof(double);

    if (d_E_point) CUDA_CHECK(cudaFree(d_E_point));
    if (d_contract_coef) CUDA_CHECK(cudaFree(d_contract_coef));
    if (d_contract_idxs) CUDA_CHECK(cudaFree(d_contract_idxs));
    if (d_particle_index) CUDA_CHECK(cudaFree(d_particle_index));
    if (d_G_point) { CUDA_CHECK(cudaFree(d_G_point)); d_G_point = nullptr; }
    if (d_contract_coef_Q) { CUDA_CHECK(cudaFree(d_contract_coef_Q)); d_contract_coef_Q = nullptr; }

    CUDA_CHECK(cudaMalloc(&d_E_point, size_epoint_bytes));
    CUDA_CHECK(cudaMemset(d_E_point, 0, size_epoint_bytes));

    CUDA_CHECK(cudaMalloc(&d_contract_coef, size_coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_contract_idxs, size_idxs_bytes));
    CUDA_CHECK(cudaMalloc(&d_particle_index, size_idx_bytes));

    if (solve_quadrupoles) {
        CUDA_CHECK(cudaMalloc(&d_G_point, num_field_points * 5 * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_G_point, 0, num_field_points * 5 * 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_contract_coef_Q, num_contract * sizeof(double)));
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_contract + threadsPerBlock - 1) / threadsPerBlock;

    // In dipole solver, x_field is x_part, and field point radii are radii
    contract_precalcs_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
        d_x_field, d_y_field, d_z_field,
        d_radii, // using particle radii as field points are particle centers
        d_offset, d_offsetxyz,
        d_contract_coef, d_contract_idxs, d_particle_index,
        num_field_points, num_particles, num_offsets,
        grid_spacing[0], grid_spacing[1], grid_spacing[2],
        num_grid[0], num_grid[1], num_grid[2],
        eta_scalar, xi
    );
    CUDA_CHECK(cudaGetLastError());

    if (solve_quadrupoles) {
        double const_factor = std::pow(2.0 * xi * xi / 3.14159265358979323846, 1.5) * grid_spacing[0] * grid_spacing[1] * grid_spacing[2];
        contract_precalcs_kernel_point<<<blocksPerGrid, threadsPerBlock>>>(
            d_x_field, d_y_field, d_z_field,
            d_offset, d_offsetxyz,
            d_contract_coef_Q,
            num_field_points, num_offsets,
            grid_spacing[0], grid_spacing[1], grid_spacing[2],
            eta_scalar, xi,
            const_factor
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Polydisperse_Ewald_Electric_Field::realSpacePrecalcs() {
    computeNeighborList(128);

    size_t num_pairs = neighbor_list->get_num_pairs();

    if (d_perp) CUDA_CHECK(cudaFree(d_perp));
    if (d_para) CUDA_CHECK(cudaFree(d_para));
    if (d_perp_Q) { cudaFree(d_perp_Q); d_perp_Q = nullptr; }
    if (d_para_Q) { cudaFree(d_para_Q); d_para_Q = nullptr; }
    if (d_Q3) { cudaFree(d_Q3); d_Q3 = nullptr; }
    if (d_G1) { cudaFree(d_G1); d_G1 = nullptr; }
    if (d_G2) { cudaFree(d_G2); d_G2 = nullptr; }
    if (d_G3) { cudaFree(d_G3); d_G3 = nullptr; }
    if (d_G4) { cudaFree(d_G4); d_G4 = nullptr; }

    if (num_pairs > 0) {
        CUDA_CHECK(cudaMalloc(&d_perp, num_pairs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_para, num_pairs * sizeof(double)));

        if (solve_quadrupoles) {
            CUDA_CHECK(cudaMalloc(&d_perp_Q, num_pairs * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_para_Q, num_pairs * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_Q3, num_pairs * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_G1, num_pairs * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_G2, num_pairs * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_G3, num_pairs * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_G4, num_pairs * sizeof(double)));
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

        real_space_precalcs_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
            d_x_part, d_y_part, d_z_part,
            d_x_field, d_y_field, d_z_field,
            d_radius_idx, d_col_ind, num_unique_radii,
            neighbor_list->get_list(), neighbor_list->get_counts(), neighbor_list->get_offsets(),
            d_r_table, d_field_dip_1, d_field_dip_2,
            table_size,
            d_perp, d_para,
            num_particles,
            neighbor_list->get_max_neighbors(),
            box_x, box_y, box_z,
            rc,
            solve_quadrupoles,
            d_field_quad_1, d_field_quad_2, d_field_quad_3,
            d_grad_quad_1, d_grad_quad_2, d_grad_quad_3, d_grad_quad_4,
            d_perp_Q, d_para_Q, d_Q3,
            d_G1, d_G2, d_G3, d_G4
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        d_perp = nullptr;
        d_para = nullptr;
    }
}

void Polydisperse_Ewald_Electric_Field::spread(double* d_fE_grid) {
    if (num_spread == 0 || d_spread_coef == nullptr || d_spread_idxs == nullptr || d_fE_grid == nullptr) {
        throw std::runtime_error("spread: Buffers/Precalcs are not allocated.");
    }

    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    CUDA_CHECK(cudaMemset(d_fE_grid, 0, grid_voxels * 2 * sizeof(double)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_spread + threadsPerBlock - 1) / threadsPerBlock;

    spread_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock, 8192>>>(
        d_dipoles,
        d_spread_coef, d_spread_idxs,
        d_fE_grid,
        num_spread, num_offsets,
        num_grid[0], num_grid[1], num_grid[2]
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Polydisperse_Ewald_Electric_Field::scale(double* d_fE_grid) {
    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    if (grid_voxels == 0 || d_scale_coef == nullptr || d_fE_grid == nullptr) {
        throw std::runtime_error("scale: Buffers/Precalcs are not allocated.");
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (grid_voxels + threadsPerBlock - 1) / threadsPerBlock;

    if (solve_quadrupoles) {
        scale_kernel_joint_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
            d_fE_grid,
            d_fGs_grid,
            d_scale_coef,
            d_scale_coef_Q_imag,
            d_scale_coef_GP_imag,
            d_scale_coef_GQ_real,
            d_Qfactor,
            d_Qfactor_dot,
            grid_voxels
        );
    } else {
        scale_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
            d_fE_grid,
            d_scale_coef,
            grid_voxels
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Polydisperse_Ewald_Electric_Field::contract(double* d_E_point, const double* d_Es_grid) {
    if (num_contract == 0 || d_contract_coef == nullptr || d_contract_idxs == nullptr || d_particle_index == nullptr || d_E_point == nullptr) {
        throw std::runtime_error("contract: Buffers/Precalcs are not allocated.");
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_field_points + threadsPerBlock - 1) / threadsPerBlock;

    double prod_h = grid_spacing[0] * grid_spacing[1] * grid_spacing[2];

    contract_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
        d_Es_grid,
        d_contract_coef, d_contract_idxs,
        d_E_point,
        num_field_points, num_offsets,
        num_grid[0], num_grid[1], num_grid[2],
        prod_h
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Polydisperse_Ewald_Electric_Field::realSpace(double* d_E_point) {
    if (num_particles == 0 || d_E_point == nullptr) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    if (mode == FieldCalcMode::SOLVER_AX) {
        if (solve_quadrupoles) {
            real_space_self_kernel_polydisperse_joint<<<blocksPerGrid, threadsPerBlock>>>(
                d_dipoles, d_self_coef_r, d_self_coef_i,
                d_radius_idx, d_self_perp_uniq,
                d_quad_idxs, d_quad_map,
                d_self_G2_uniq,
                d_radii,
                d_E_point,
                num_particles, num_quads,
                solve_quadrupoles,
                mode
            );
        } else {
            real_space_self_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
                d_dipoles, d_self_coef_r, d_self_coef_i,
                d_radius_idx, d_self_perp_uniq,
                d_E_point,
                num_particles,
                mode
            );
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t num_pairs = neighbor_list ? neighbor_list->get_num_pairs() : 0;
    if (num_pairs > 0) {
        if (solve_quadrupoles) {
            real_space_neighbor_kernel_polydisperse_joint<<<blocksPerGrid, threadsPerBlock>>>(
                d_x_part, d_y_part, d_z_part,
                d_x_field, d_y_field, d_z_field,
                d_dipoles,
                neighbor_list->get_list(), neighbor_list->get_counts(), neighbor_list->get_offsets(),
                d_quad_idxs, d_quad_map,
                d_perp, d_para,
                d_perp_Q, d_para_Q, d_Q3,
                d_G1, d_G2, d_G3, d_G4,
                d_E_point,
                num_particles, num_quads,
                neighbor_list->get_max_neighbors(),
                box_x, box_y, box_z,
                rc,
                solve_quadrupoles,
                mode
            );
        } else {
            real_space_neighbor_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
                d_x_part, d_y_part, d_z_part,
                d_x_field, d_y_field, d_z_field,
                d_dipoles,
                neighbor_list->get_list(), neighbor_list->get_counts(), neighbor_list->get_offsets(),
                d_perp, d_para,
                d_E_point,
                num_particles,
                neighbor_list->get_max_neighbors(),
                box_x, box_y, box_z,
                rc,
                mode
            );
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

__global__ void negate_vector_kernel_poly(double* vec, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = -vec[idx];
    }
}

void Polydisperse_Ewald_Electric_Field::electricField() {
    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];

    spread(d_fE_grid);

    if (solve_quadrupoles && num_quads > 0 && d_fG_grid != nullptr) {
        CUDA_CHECK(cudaMemset(d_fG_grid, 0, grid_voxels * 5 * 2 * sizeof(double)));

        size_t total_spread_Q = num_quads * num_offsets;
        int threadsPerBlock = 256;
        int blocksPerGrid = (total_spread_Q + threadsPerBlock - 1) / threadsPerBlock;

        spread_quadrupoles_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock, 24576>>>(
            d_dipoles,
            d_quad_idxs,
            d_spread_coef_Q, d_spread_idxs,
            d_fG_grid,
            num_quads, num_particles, num_offsets,
            num_grid[0], num_grid[1], num_grid[2]
        );
        CUDA_CHECK(cudaGetLastError());

        cufftResult plan_res_G = cufftExecZ2Z((cufftHandle)fft_plan_G,
                                             (cufftDoubleComplex*)d_fG_grid,
                                             (cufftDoubleComplex*)d_fG_grid,
                                             CUFFT_FORWARD);
        if (plan_res_G != CUFFT_SUCCESS) {
            throw std::runtime_error("cuFFT forward execution G failed with code: " + std::to_string(plan_res_G));
        }
    }

    cufftResult plan_res = cufftExecZ2Z((cufftHandle)fft_plan,
                                       (cufftDoubleComplex*)d_fE_grid,
                                       (cufftDoubleComplex*)d_fE_grid,
                                       CUFFT_FORWARD);
    if (plan_res != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT forward execution failed with code: " + std::to_string(plan_res));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int threadsPerBlock = 256;
    int blocksPerGridShift = (grid_voxels + threadsPerBlock - 1) / threadsPerBlock;
    
    fftshift_3d_kernel_polydisperse<<<blocksPerGridShift, threadsPerBlock>>>(
        d_fE_grid, d_fEs_grid, num_grid[0], num_grid[1], num_grid[2], 1
    );
    if (solve_quadrupoles && num_quads > 0 && d_fG_grid != nullptr) {
        fftshift_3d_kernel_polydisperse<<<blocksPerGridShift, threadsPerBlock>>>(
            d_fG_grid, d_fGs_grid, num_grid[0], num_grid[1], num_grid[2], 5
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    scale(d_fEs_grid);

    ifftshift_3d_kernel_polydisperse<<<blocksPerGridShift, threadsPerBlock>>>(
        d_fEs_grid, d_fE_grid, num_grid[0], num_grid[1], num_grid[2], 1
    );
    if (solve_quadrupoles && num_quads > 0 && d_fG_grid != nullptr) {
        ifftshift_3d_kernel_polydisperse<<<blocksPerGridShift, threadsPerBlock>>>(
            d_fGs_grid, d_fG_grid, num_grid[0], num_grid[1], num_grid[2], 5
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    plan_res = cufftExecZ2Z((cufftHandle)fft_plan,
                           (cufftDoubleComplex*)d_fE_grid,
                           (cufftDoubleComplex*)d_fE_grid,
                           CUFFT_INVERSE);
    if (plan_res != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT inverse execution failed with code: " + std::to_string(plan_res));
    }

    if (solve_quadrupoles && num_quads > 0 && d_fG_grid != nullptr) {
        cufftResult plan_res_G = cufftExecZ2Z((cufftHandle)fft_plan_G,
                                             (cufftDoubleComplex*)d_fG_grid,
                                             (cufftDoubleComplex*)d_fG_grid,
                                             CUFFT_INVERSE);
        if (plan_res_G != CUFFT_SUCCESS) {
            throw std::runtime_error("cuFFT inverse execution G failed with code: " + std::to_string(plan_res_G));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (d_E_point == nullptr) {
        throw std::runtime_error("electricField: d_E_point has not been allocated (contractPrecalcs not run).");
    }
    size_t size_epoint_bytes = (num_field_points * 3 + num_quads * 5) * 2 * sizeof(double);
    CUDA_CHECK(cudaMemset(d_E_point, 0, size_epoint_bytes));

    contract(d_E_point, d_fE_grid);

    if (solve_quadrupoles && num_quads > 0 && d_G_point != nullptr) {
        CUDA_CHECK(cudaMemset(d_G_point, 0, num_field_points * 5 * 2 * sizeof(double)));

        int contract_threads = 256;
        int contract_blocks = (num_field_points + contract_threads - 1) / contract_threads;
        contract_kernel_G_polydisperse<<<contract_blocks, contract_threads>>>(
            d_fG_grid,
            d_contract_idxs,
            d_contract_coef_Q,
            d_G_point,
            num_field_points,
            num_offsets,
            num_grid[1],
            num_grid[2]
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int copy_threads = 256;
        int copy_blocks = (num_quads + copy_threads - 1) / copy_threads;
        copy_G_to_E_kernel_polydisperse<<<copy_blocks, copy_threads>>>(
            d_G_point,
            d_quad_idxs,
            d_E_point,
            num_quads,
            num_field_points
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (mode == FieldCalcMode::INTERACTION_FIELD) {
        size_t num_targets = (num_field_points > 0) ? num_field_points : num_particles;
        size_t num_doubles_E = (num_targets * 3 + num_quads * 5) * 2;
        if (num_doubles_E > 0) {
            int threads = 256;
            int blocks = (num_doubles_E + threads - 1) / threads;
            negate_vector_kernel_poly<<<blocks, threads>>>(d_E_point, num_doubles_E);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    realSpace(d_E_point);
}

void Polydisperse_Ewald_Electric_Field::calculate() {
    if (particles_updated || field_points_updated || d_perp == nullptr || (solve_quadrupoles && d_perp_Q == nullptr)) {
        realSpacePrecalcs();
    }
    if (field_points_updated || d_contract_coef == nullptr || (solve_quadrupoles && d_contract_coef_Q == nullptr)) {
        contractPrecalcs();
        field_points_updated = false;
    }
    if (particles_updated || d_spread_coef == nullptr || (solve_quadrupoles && d_spread_coef_Q == nullptr)) {
        spreadPrecalcs();
        particles_updated = false;
    }

    electricField();
}
