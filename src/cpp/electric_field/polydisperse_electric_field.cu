#include "polydisperse_electric_field.h"
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
// CUDA Kernels for Polydisperse Ewald Solver
// -----------------------------------------------------------------------------

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

    spread_idxs[idx] = ((gix + ox + 256) << 20) | ((giy + oy + 256) << 10) | (giz + oz + 256);

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
    double rc)
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
            int radius_idx_j = radius_idx[j];
            int col = col_ind[radius_idx_i * num_cols_unique + radius_idx_j];
            int num_cols_total = num_cols_unique * (num_cols_unique + 1) / 2;

            double p_val = interpolate_table_gpu_polydisperse(d, r_table, field_dip_1, table_size, col, num_cols_total);
            double a_val = interpolate_table_gpu_polydisperse(d, r_table, field_dip_2, table_size, col, num_cols_total);

            perp[start_idx + k] = p_val;
            para[start_idx + k] = a_val;
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
    size_t num_particles)
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
    double rc)
{
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
            double delta_x = rx / d;
            double delta_y = ry / d;
            double delta_z = rz / d;

            // neighbor dipoles
            const double2* d_dipoles_d2 = reinterpret_cast<const double2*>(d_dipoles);
            double2 dip_x = d_dipoles_d2[j * 3 + 0];
            double2 dip_y = d_dipoles_d2[j * 3 + 1];
            double2 dip_z = d_dipoles_d2[j * 3 + 2];

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

            atomicAdd(&E_point[(i * 3 + 0) * 2 + 0], contrib_x_r);
            atomicAdd(&E_point[(i * 3 + 0) * 2 + 1], contrib_x_i);

            atomicAdd(&E_point[(i * 3 + 1) * 2 + 0], contrib_y_r);
            atomicAdd(&E_point[(i * 3 + 1) * 2 + 1], contrib_y_i);

            atomicAdd(&E_point[(i * 3 + 2) * 2 + 0], contrib_z_r);
            atomicAdd(&E_point[(i * 3 + 2) * 2 + 1], contrib_z_i);
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

    double scale_factor = 1.0 / static_cast<double>(num_voxels);

    output[shifted_idx * 2 + 0] = input[idx * 2 + 0] * scale_factor;
    output[shifted_idx * 2 + 1] = input[idx * 2 + 1] * scale_factor;
}

// -----------------------------------------------------------------------------
// Polydisperse_Electric_Field Implementation
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

Polydisperse_Electric_Field::Polydisperse_Electric_Field(
    double box_x, double box_y, double box_z,
    double errortol,
    double xi,
    bool calc_inter_dipole,
    const std::vector<double>& particle_radii)
    : box_x(box_x), box_y(box_y), box_z(box_z), errortol(errortol),
      xi(xi), calc_inter_dipole(calc_inter_dipole),
      h_radii(particle_radii),
      particles_updated(false), field_points_updated(false),
      dipoles_updated(false),
      neighbor_list(std::make_unique<NeighborList>())
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
}

Polydisperse_Electric_Field::~Polydisperse_Electric_Field() {
    if (d_x_part) cudaFree(d_x_part);
    if (d_y_part) cudaFree(d_y_part);
    if (d_z_part) cudaFree(d_z_part);
    if (d_x_field) cudaFree(d_x_field);
    if (d_y_field) cudaFree(d_y_field);
    if (d_z_field) cudaFree(d_z_field);
    if (d_radii) cudaFree(d_radii);
    if (d_radius_idx) cudaFree(d_radius_idx);
    if (d_col_ind) cudaFree(d_col_ind);
    if (d_self_perp_uniq) cudaFree(d_self_perp_uniq);
    if (d_r_table) cudaFree(d_r_table);
    if (d_field_dip_1) cudaFree(d_field_dip_1);
    if (d_field_dip_2) cudaFree(d_field_dip_2);
    if (d_offset) cudaFree(d_offset);
    if (d_offsetxyz) cudaFree(d_offsetxyz);
    if (d_scale_coef) cudaFree(d_scale_coef);
    if (d_dipoles) cudaFree(d_dipoles);
    if (d_self_coef_r) cudaFree(d_self_coef_r);
    if (d_self_coef_i) cudaFree(d_self_coef_i);
    if (d_spread_coef) cudaFree(d_spread_coef);
    if (d_spread_idxs) cudaFree(d_spread_idxs);
    if (d_E_point) cudaFree(d_E_point);
    if (d_particle_index) cudaFree(d_particle_index);
    if (d_contract_coef) cudaFree(d_contract_coef);
    if (d_contract_idxs) cudaFree(d_contract_idxs);
    if (d_perp) cudaFree(d_perp);
    if (d_para) cudaFree(d_para);
    if (d_fE_grid) cudaFree(d_fE_grid);
    if (d_fEs_grid) cudaFree(d_fEs_grid);
    if (fft_plan) cufftDestroy((cufftHandle)fft_plan);
}

void Polydisperse_Electric_Field::computeNeighborList(int max_neighbors_per_particle) {
    neighbor_list->build(
        d_x_part, d_y_part, d_z_part,
        d_x_field, d_y_field, d_z_field,
        num_particles, num_field_points,
        box_x, box_y, box_z,
        rc, calc_inter_dipole,
        max_neighbors_per_particle
    );
}

void Polydisperse_Electric_Field::computeRealSpaceTables() {
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
}

void Polydisperse_Electric_Field::computePrecalculations() {
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
                    continue;
                }

                double kx_val = Kx[ix];
                double ky_val = Ky[iy];
                double kz_val = Kz[iz];

                double ksqsm = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;

                double exp_part = std::exp(-(1.0 - eta_scalar) * ksqsm / (4.0 * xi * xi));
                host_scale_coef[linear_idx] = exp_part / ksqsm;
            }
        }
    }

    if (d_offset) CUDA_CHECK(cudaFree(d_offset));
    if (d_offsetxyz) CUDA_CHECK(cudaFree(d_offsetxyz));
    if (d_scale_coef) CUDA_CHECK(cudaFree(d_scale_coef));

    CUDA_CHECK(cudaMalloc(&d_offset, num_offsets * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsetxyz, num_offsets * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scale_coef, grid_voxels * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_offset, host_offset.data(), num_offsets * 3 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsetxyz, host_offsetxyz.data(), num_offsets * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_coef, host_scale_coef.data(), grid_voxels * sizeof(double), cudaMemcpyHostToDevice));
}

void Polydisperse_Electric_Field::updateParticleCoordinates(
    const std::vector<double>& x_part,
    const std::vector<double>& y_part,
    const std::vector<double>& z_part)
{
    size_t prev_num_particles = num_particles;
    num_particles = x_part.size();
    size_t size_bytes = num_particles * sizeof(double);
    size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);

    if (num_particles != prev_num_particles || d_dipoles == nullptr) {
        if (d_x_part) cudaFree(d_x_part);
        if (d_y_part) cudaFree(d_y_part);
        if (d_z_part) cudaFree(d_z_part);
        if (d_dipoles) cudaFree(d_dipoles);
        if (d_self_coef_r) { cudaFree(d_self_coef_r); d_self_coef_r = nullptr; }
        if (d_self_coef_i) { cudaFree(d_self_coef_i); d_self_coef_i = nullptr; }

        CUDA_CHECK(cudaMalloc(&d_x_part, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_y_part, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_z_part, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_dipoles, size_dipoles_bytes));
        CUDA_CHECK(cudaMemset(d_dipoles, 0, size_dipoles_bytes));
    }

    CUDA_CHECK(cudaMemcpy(d_x_part, x_part.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_part, y_part.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_part, z_part.data(), size_bytes, cudaMemcpyHostToDevice));

    particles_updated = true;

    if (calc_inter_dipole) {
        if (num_particles != prev_num_particles || d_x_field == nullptr) {
            num_field_points = num_particles;
            if (d_x_field) cudaFree(d_x_field);
            if (d_y_field) cudaFree(d_y_field);
            if (d_z_field) cudaFree(d_z_field);

            CUDA_CHECK(cudaMalloc(&d_x_field, size_bytes));
            CUDA_CHECK(cudaMalloc(&d_y_field, size_bytes));
            CUDA_CHECK(cudaMalloc(&d_z_field, size_bytes));
        }

        CUDA_CHECK(cudaMemcpy(d_x_field, d_x_part, size_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_field, d_y_part, size_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_z_field, d_z_part, size_bytes, cudaMemcpyDeviceToDevice));

        field_points_updated = true;
    }
}

void Polydisperse_Electric_Field::updateFieldCoordinates(
    const std::vector<double>& x_field,
    const std::vector<double>& y_field,
    const std::vector<double>& z_field)
{
    num_field_points = x_field.size();
    size_t size_bytes = num_field_points * sizeof(double);

    if (d_x_field) cudaFree(d_x_field);
    if (d_y_field) cudaFree(d_y_field);
    if (d_z_field) cudaFree(d_z_field);

    CUDA_CHECK(cudaMalloc(&d_x_field, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_y_field, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_z_field, size_bytes));

    CUDA_CHECK(cudaMemcpy(d_x_field, x_field.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_field, y_field.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_field, z_field.data(), size_bytes, cudaMemcpyHostToDevice));

    field_points_updated = true;
}

void Polydisperse_Electric_Field::updateDipoles(
    const std::vector<double>& dip_x,
    const std::vector<double>& dip_y,
    const std::vector<double>& dip_z)
{
    size_t size_bytes = num_particles * 3 * 2 * sizeof(double);
    if (d_dipoles == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_dipoles, size_bytes));
    }
    CUDA_CHECK(cudaMemset(d_dipoles, 0, size_bytes));

    std::vector<double> host_dip(num_particles * 3 * 2, 0.0);
    for (size_t i = 0; i < num_particles; ++i) {
        host_dip[(i * 3 + 0) * 2 + 0] = dip_x[i];
        host_dip[(i * 3 + 1) * 2 + 0] = dip_y[i];
        host_dip[(i * 3 + 2) * 2 + 0] = dip_z[i];
    }
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dip.data(), size_bytes, cudaMemcpyHostToDevice));
    dipoles_updated = true;
}

void Polydisperse_Electric_Field::updateDipolesComplex(
    const std::vector<double>& dip_xr, const std::vector<double>& dip_xi,
    const std::vector<double>& dip_yr, const std::vector<double>& dip_yi,
    const std::vector<double>& dip_zr, const std::vector<double>& dip_zi)
{
    size_t size_bytes = num_particles * 3 * 2 * sizeof(double);
    if (d_dipoles == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_dipoles, size_bytes));
    }

    std::vector<double> host_dip(num_particles * 3 * 2, 0.0);
    for (size_t i = 0; i < num_particles; ++i) {
        host_dip[(i * 3 + 0) * 2 + 0] = dip_xr[i];
        host_dip[(i * 3 + 0) * 2 + 1] = dip_xi[i];
        host_dip[(i * 3 + 1) * 2 + 0] = dip_yr[i];
        host_dip[(i * 3 + 1) * 2 + 1] = dip_yi[i];
        host_dip[(i * 3 + 2) * 2 + 0] = dip_zr[i];
        host_dip[(i * 3 + 2) * 2 + 1] = dip_zi[i];
    }
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dip.data(), size_bytes, cudaMemcpyHostToDevice));
    dipoles_updated = true;
}

void Polydisperse_Electric_Field::setSelfCoef(const std::vector<double>& self_coef_r, const std::vector<double>& self_coef_i) {
    if (d_self_coef_r == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_self_coef_r, num_particles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, num_particles * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(d_self_coef_r, self_coef_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, self_coef_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
}

void Polydisperse_Electric_Field::setSelfCoef(double val_r, double val_i) {
    if (d_self_coef_r == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_self_coef_r, num_particles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, num_particles * sizeof(double)));
    }
    std::vector<double> host_sc_r(num_particles, val_r);
    std::vector<double> host_sc_i(num_particles, val_i);
    CUDA_CHECK(cudaMemcpy(d_self_coef_r, host_sc_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, host_sc_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
}

void Polydisperse_Electric_Field::spreadPrecalcs() {
    if (num_particles == 0) return;

    num_spread = num_particles * num_offsets;
    size_t size_coef_bytes = num_spread * 3 * sizeof(double);
    size_t size_idxs_bytes = num_spread * sizeof(int);

    if (d_spread_coef) CUDA_CHECK(cudaFree(d_spread_coef));
    if (d_spread_idxs) CUDA_CHECK(cudaFree(d_spread_idxs));

    CUDA_CHECK(cudaMalloc(&d_spread_coef, size_coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_spread_idxs, size_idxs_bytes));

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
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Polydisperse_Electric_Field::contractPrecalcs() {
    if (num_field_points == 0) return;

    num_contract = num_field_points * num_offsets;
    size_t size_coef_bytes = num_contract * 3 * sizeof(double);
    size_t size_idxs_bytes = num_contract * sizeof(int);
    size_t size_idx_bytes = num_contract * sizeof(int);
    size_t size_epoint_bytes = num_field_points * 3 * 2 * sizeof(double);

    if (d_E_point) CUDA_CHECK(cudaFree(d_E_point));
    if (d_contract_coef) CUDA_CHECK(cudaFree(d_contract_coef));
    if (d_contract_idxs) CUDA_CHECK(cudaFree(d_contract_idxs));
    if (d_particle_index) CUDA_CHECK(cudaFree(d_particle_index));

    CUDA_CHECK(cudaMalloc(&d_E_point, size_epoint_bytes));
    CUDA_CHECK(cudaMemset(d_E_point, 0, size_epoint_bytes));

    CUDA_CHECK(cudaMalloc(&d_contract_coef, size_coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_contract_idxs, size_idxs_bytes));
    CUDA_CHECK(cudaMalloc(&d_particle_index, size_idx_bytes));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_contract + threadsPerBlock - 1) / threadsPerBlock;

    // In dipole solver, x_field is x_part, and field point radii are radii
    contract_precalcs_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
        d_x_field, d_y_field, d_z_field,
        d_radii, // using particle radii as field points are particle centers
        d_offset, d_offsetxyz,
        d_contract_coef, d_contract_idxs, d_particle_index,
        num_field_points, num_offsets,
        grid_spacing[0], grid_spacing[1], grid_spacing[2],
        num_grid[0], num_grid[1], num_grid[2],
        eta_scalar, xi
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Polydisperse_Electric_Field::realSpacePrecalcs() {
    computeNeighborList(128);

    size_t num_pairs = neighbor_list->get_num_pairs();

    if (d_perp) CUDA_CHECK(cudaFree(d_perp));
    if (d_para) CUDA_CHECK(cudaFree(d_para));

    if (num_pairs > 0) {
        CUDA_CHECK(cudaMalloc(&d_perp, num_pairs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_para, num_pairs * sizeof(double)));

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
            rc
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        d_perp = nullptr;
        d_para = nullptr;
    }
}

void Polydisperse_Electric_Field::spread(double* d_fE_grid) {
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

void Polydisperse_Electric_Field::scale(double* d_fE_grid) {
    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    if (grid_voxels == 0 || d_scale_coef == nullptr || d_fE_grid == nullptr) {
        throw std::runtime_error("scale: Buffers/Precalcs are not allocated.");
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (grid_voxels + threadsPerBlock - 1) / threadsPerBlock;

    scale_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
        d_fE_grid,
        d_scale_coef,
        grid_voxels
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Polydisperse_Electric_Field::contract(double* d_E_point, const double* d_Es_grid) {
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

void Polydisperse_Electric_Field::realSpace(double* d_E_point) {
    if (num_particles == 0 || d_E_point == nullptr) return;

    if (calc_inter_dipole) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

        real_space_self_kernel_polydisperse<<<blocksPerGrid, threadsPerBlock>>>(
            d_dipoles, d_self_coef_r, d_self_coef_i,
            d_radius_idx, d_self_perp_uniq,
            d_E_point,
            num_particles
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t num_pairs = neighbor_list ? neighbor_list->get_num_pairs() : 0;
    if (num_pairs > 0) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

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
            rc
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void Polydisperse_Electric_Field::electricField() {
    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];

    spread(d_fE_grid);

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
    fftshift_3d_kernel_scalar<<<blocksPerGridShift, threadsPerBlock>>>(
        d_fE_grid, d_fEs_grid, num_grid[0], num_grid[1], num_grid[2]
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    scale(d_fEs_grid);

    ifftshift_3d_kernel_scalar<<<blocksPerGridShift, threadsPerBlock>>>(
        d_fEs_grid, d_fE_grid, num_grid[0], num_grid[1], num_grid[2]
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    plan_res = cufftExecZ2Z((cufftHandle)fft_plan,
                           (cufftDoubleComplex*)d_fE_grid,
                           (cufftDoubleComplex*)d_fE_grid,
                           CUFFT_INVERSE);
    if (plan_res != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT inverse execution failed with code: " + std::to_string(plan_res));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (d_E_point == nullptr) {
        throw std::runtime_error("electricField: d_E_point has not been allocated (contractPrecalcs not run).");
    }
    size_t size_epoint_bytes = num_field_points * 3 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemset(d_E_point, 0, size_epoint_bytes));

    contract(d_E_point, d_fE_grid);

    realSpace(d_E_point);
}

void Polydisperse_Electric_Field::calculate() {
    if (particles_updated || field_points_updated || d_perp == nullptr) {
        realSpacePrecalcs();
    }
    if (field_points_updated || d_contract_coef == nullptr) {
        contractPrecalcs();
        field_points_updated = false;
    }
    if (particles_updated || d_spread_coef == nullptr) {
        spreadPrecalcs();
        particles_updated = false;
    }

    electricField();
}
