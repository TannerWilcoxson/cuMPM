#include "neighbor_list.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
      if (abort) exit(code);
   }
}

// CUDA kernel to compute cell assignment for each field point
__global__ void assign_cells_kernel(
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
    int* __restrict__ particle_cell_ids,
    int* __restrict__ particle_ids,
    size_t num_field_points,
    double box_x, double box_y, double box_z,
    double dx_cell, double dy_cell, double dz_cell,
    int N_cx, int N_cy, int N_cz
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_field_points) return;

    double xj = x_field[j];
    double yj = y_field[j];
    double zj = z_field[j];

    // periodic wrap coordinates to [0, box)
    double wx = xj - box_x * floor(xj / box_x);
    double wy = yj - box_y * floor(yj / box_y);
    double wz = zj - box_z * floor(zj / box_z);

    int cx = (int)floor(wx / dx_cell);
    int cy = (int)floor(wy / dy_cell);
    int cz = (int)floor(wz / dz_cell);

    if (cx < 0) cx = 0; if (cx >= N_cx) cx = N_cx - 1;
    if (cy < 0) cy = 0; if (cy >= N_cy) cy = N_cy - 1;
    if (cz < 0) cz = 0; if (cz >= N_cz) cz = N_cz - 1;

    int cell_id = cx + cy * N_cx + cz * N_cx * N_cy;

    particle_cell_ids[j] = cell_id;
    particle_ids[j] = j;
}

// CUDA kernel to identify the starting and ending index of each cell in the sorted arrays
__global__ void find_cell_bounds_kernel(
    const int* __restrict__ sorted_cell_ids,
    int* __restrict__ cell_starts,
    int* __restrict__ cell_ends,
    size_t num_field_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_field_points) return;

    int curr_cell = sorted_cell_ids[idx];
    if (idx == 0) {
        cell_starts[curr_cell] = 0;
    } else {
        int prev_cell = sorted_cell_ids[idx - 1];
        if (curr_cell != prev_cell) {
            cell_ends[prev_cell] = idx;
            cell_starts[curr_cell] = idx;
        }
    }
    if (idx == num_field_points - 1) {
        cell_ends[curr_cell] = num_field_points;
    }
}

// CUDA kernel to compute neighbor list using the constructed cell lists
__global__ void compute_neighbor_list_cell_kernel(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_ends,
    const int* __restrict__ sorted_particle_ids,
    int* __restrict__ neighbor_list,
    int* __restrict__ neighbor_counts,
    size_t num_particles,
    double box_x, double box_y, double box_z,
    double dx_cell, double dy_cell, double dz_cell,
    int N_cx, int N_cy, int N_cz,
    double cutoff,
    int max_neighbors,
    bool calc_inter_dipole
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    double xi = x_part[i];
    double yi = y_part[i];
    double zi = z_part[i];
    double cutoff_sq = cutoff * cutoff;

    // Wrap query particle coordinates to [0, box)
    double wx = xi - box_x * floor(xi / box_x);
    double wy = yi - box_y * floor(yi / box_y);
    double wz = zi - box_z * floor(zi / box_z);

    int cx = (int)floor(wx / dx_cell);
    int cy = (int)floor(wy / dy_cell);
    int cz = (int)floor(wz / dz_cell);

    if (cx < 0) cx = 0; if (cx >= N_cx) cx = N_cx - 1;
    if (cy < 0) cy = 0; if (cy >= N_cy) cy = N_cy - 1;
    if (cz < 0) cz = 0; if (cz >= N_cz) cz = N_cz - 1;

    int count = 0;

    // Loop over the 27 neighboring cells (wrap index boundaries periodically)
    for (int dz = -1; dz <= 1; ++dz) {
        int ncz = cz + dz;
        if (ncz < 0) ncz += N_cz;
        else if (ncz >= N_cz) ncz -= N_cz;

        for (int dy = -1; dy <= 1; ++dy) {
            int ncy = cy + dy;
            if (ncy < 0) ncy += N_cy;
            else if (ncy >= N_cy) ncy -= N_cy;

            for (int dx = -1; dx <= 1; ++dx) {
                int ncx = cx + dx;
                if (ncx < 0) ncx += N_cx;
                else if (ncx >= N_cx) ncx -= N_cx;

                int cell_idx = ncx + ncy * N_cx + ncz * N_cx * N_cy;
                int start = cell_starts[cell_idx];
                if (start != -1) {
                    int end = cell_ends[cell_idx];
                    for (int p_idx = start; p_idx < end; ++p_idx) {
                        int j = sorted_particle_ids[p_idx];
                        if (calc_inter_dipole && (i == j)) continue;

                        double dx_val = xi - x_field[j];
                        double dy_val = yi - y_field[j];
                        double dz_val = zi - z_field[j];

                        // Periodic boundary minimum image convention
                        if (box_x > 0.0) dx_val -= box_x * round(dx_val / box_x);
                        if (box_y > 0.0) dy_val -= box_y * round(dy_val / box_y);
                        if (box_z > 0.0) dz_val -= box_z * round(dz_val / box_z);

                        double dist_sq = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;
                        if (dist_sq <= cutoff_sq) {
                            if (count < max_neighbors) {
                                neighbor_list[i * max_neighbors + count] = j;
                            }
                            count++;
                        }
                    }
                }
            }
        }
    }
    neighbor_counts[i] = count;
}

// CUDA kernel to compute neighbor list under periodic boundary conditions (brute force fallback)
__global__ void compute_neighbor_list_kernel(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
    int* __restrict__ neighbor_list,
    int* __restrict__ neighbor_counts,
    size_t num_particles,
    size_t num_field_points,
    double box_x,
    double box_y,
    double box_z,
    double cutoff,
    int max_neighbors,
    bool calc_inter_dipole) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    double xi = x_part[i];
    double yi = y_part[i];
    double zi = z_part[i];
    double cutoff_sq = cutoff * cutoff;

    int count = 0;
    for (int j = 0; j < num_field_points; ++j) {
        if (calc_inter_dipole && (i == j)) continue;

        double dx = xi - x_field[j];
        double dy = yi - y_field[j];
        double dz = zi - z_field[j];

        // Apply periodic boundary conditions (minimum image convention)
        if (box_x > 0.0) dx -= box_x * round(dx / box_x);
        if (box_y > 0.0) dy -= box_y * round(dy / box_y);
        if (box_z > 0.0) dz -= box_z * round(dz / box_z);

        double dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq <= cutoff_sq) {
            if (count < max_neighbors) {
                neighbor_list[i * max_neighbors + count] = j;
            }
            count++;
        }
    }
    neighbor_counts[i] = count;
}

void NeighborList::free_buffers() {
    if (d_neighbor_list) { cudaFree(d_neighbor_list); d_neighbor_list = nullptr; }
    if (d_neighbor_counts) { cudaFree(d_neighbor_counts); d_neighbor_counts = nullptr; }
    if (d_particle_offsets) { cudaFree(d_particle_offsets); d_particle_offsets = nullptr; }
    if (d_particle_cell_ids) { cudaFree(d_particle_cell_ids); d_particle_cell_ids = nullptr; }
    if (d_particle_ids) { cudaFree(d_particle_ids); d_particle_ids = nullptr; }
    if (d_cell_starts) { cudaFree(d_cell_starts); d_cell_starts = nullptr; }
    if (d_cell_ends) { cudaFree(d_cell_ends); d_cell_ends = nullptr; }
    num_particles_allocated = 0;
    num_field_points_allocated = 0;
    num_cells_allocated = 0;
}

NeighborList::~NeighborList() {
    free_buffers();
}

void NeighborList::build(
    const double* d_x_part, const double* d_y_part, const double* d_z_part,
    const double* d_x_field, const double* d_y_field, const double* d_z_field,
    size_t num_particles, size_t num_field_points,
    double box_x, double box_y, double box_z,
    double rc, bool calc_inter_dipole,
    int initial_max_neighbors
) {
    if (num_particles == 0) {
        free_buffers();
        num_pairs = 0;
        num_particles_allocated = 0;
        max_neighbors = 0;
        return;
    }

    int target_max_neighbors = max_neighbors > 0 ? max_neighbors : initial_max_neighbors;

    // 1. Determine cell grid division
    int N_cx = (box_x > 0.0 && rc > 0.0) ? (int)floor(box_x / rc) : 0;
    int N_cy = (box_y > 0.0 && rc > 0.0) ? (int)floor(box_y / rc) : 0;
    int N_cz = (box_z > 0.0 && rc > 0.0) ? (int)floor(box_z / rc) : 0;
    int N_cells = N_cx * N_cy * N_cz;

    bool use_cell_list = (N_cx >= 3 && N_cy >= 3 && N_cz >= 3);

    // 2. Allocate or reallocate basic neighbor list buffers if size changes
    bool need_base_realloc = (num_particles != num_particles_allocated) || 
                             (target_max_neighbors != max_neighbors) ||
                             (d_neighbor_list == nullptr) ||
                             (d_neighbor_counts == nullptr) ||
                             (d_particle_offsets == nullptr);

    if (need_base_realloc) {
        if (d_neighbor_list) cudaFree(d_neighbor_list);
        if (d_neighbor_counts) cudaFree(d_neighbor_counts);
        if (d_particle_offsets) cudaFree(d_particle_offsets);

        max_neighbors = target_max_neighbors;
        num_particles_allocated = num_particles;
        
        CUDA_CHECK(cudaMalloc(&d_neighbor_list, num_particles * max_neighbors * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neighbor_counts, num_particles * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_particle_offsets, num_particles * sizeof(int)));
    }

    // Initialize counts to 0
    CUDA_CHECK(cudaMemset(d_neighbor_counts, 0, num_particles * sizeof(int)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    // Helper lambda to launch neighbor list search
    auto launch_search = [&](int max_nb) {
        if (use_cell_list) {
            // Allocate cell list buffers if needed
            if (d_particle_cell_ids == nullptr || num_field_points != num_field_points_allocated) {
                if (d_particle_cell_ids) cudaFree(d_particle_cell_ids);
                if (d_particle_ids) cudaFree(d_particle_ids);
                num_field_points_allocated = num_field_points;
                CUDA_CHECK(cudaMalloc(&d_particle_cell_ids, num_field_points * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_particle_ids, num_field_points * sizeof(int)));
            }

            if (d_cell_starts == nullptr || N_cells != num_cells_allocated) {
                if (d_cell_starts) cudaFree(d_cell_starts);
                if (d_cell_ends) cudaFree(d_cell_ends);
                num_cells_allocated = N_cells;
                CUDA_CHECK(cudaMalloc(&d_cell_starts, N_cells * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_cell_ends, N_cells * sizeof(int)));
            }

            // Cell list assignment
            double dx_cell = box_x / N_cx;
            double dy_cell = box_y / N_cy;
            double dz_cell = box_z / N_cz;

            int blocksField = (num_field_points + threadsPerBlock - 1) / threadsPerBlock;
            assign_cells_kernel<<<blocksField, threadsPerBlock>>>(
                d_x_field, d_y_field, d_z_field,
                d_particle_cell_ids, d_particle_ids,
                num_field_points,
                box_x, box_y, box_z,
                dx_cell, dy_cell, dz_cell,
                N_cx, N_cy, N_cz
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Sort using Thrust
            thrust::device_ptr<int> cell_ids_ptr(d_particle_cell_ids);
            thrust::device_ptr<int> particle_ids_ptr(d_particle_ids);
            thrust::sort_by_key(cell_ids_ptr, cell_ids_ptr + num_field_points, particle_ids_ptr);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Reset cell start and end offsets to -1
            CUDA_CHECK(cudaMemset(d_cell_starts, -1, N_cells * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_cell_ends, -1, N_cells * sizeof(int)));

            // Find cell boundaries
            find_cell_bounds_kernel<<<blocksField, threadsPerBlock>>>(
                d_particle_cell_ids,
                d_cell_starts, d_cell_ends,
                num_field_points
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Launch Cell List Neighbor Search
            compute_neighbor_list_cell_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_x_part, d_y_part, d_z_part,
                d_x_field, d_y_field, d_z_field,
                d_cell_starts, d_cell_ends, d_particle_ids,
                d_neighbor_list, d_neighbor_counts,
                num_particles,
                box_x, box_y, box_z,
                dx_cell, dy_cell, dz_cell,
                N_cx, N_cy, N_cz,
                rc,
                max_nb,
                calc_inter_dipole
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

        } else {
            // Fallback to Brute-Force
            compute_neighbor_list_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_x_part, d_y_part, d_z_part,
                d_x_field, d_y_field, d_z_field,
                d_neighbor_list, d_neighbor_counts,
                num_particles,
                num_field_points,
                box_x, box_y, box_z,
                rc,
                max_nb,
                calc_inter_dipole
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    };

    // First search launch with current max_neighbors
    launch_search(max_neighbors);

    // Copy counts back to check for overflow (dynamic resizing)
    std::vector<int> host_counts(num_particles);
    CUDA_CHECK(cudaMemcpy(host_counts.data(), d_neighbor_counts, num_particles * sizeof(int), cudaMemcpyDeviceToHost));

    int max_needed = 0;
    for (int c : host_counts) {
        if (c > max_needed) max_needed = c;
    }

    // If truncation occurred, reallocate and re-run search
    if (max_needed > max_neighbors) {
        CUDA_CHECK(cudaFree(d_neighbor_list));
        max_neighbors = max_needed;
        CUDA_CHECK(cudaMalloc(&d_neighbor_list, num_particles * max_neighbors * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_neighbor_counts, 0, num_particles * sizeof(int)));

        launch_search(max_neighbors);

        // Re-copy counts for offsets calculation
        CUDA_CHECK(cudaMemcpy(host_counts.data(), d_neighbor_counts, num_particles * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // Compute prefix sums (offsets) on the host and copy to device
    std::vector<int> host_offsets(num_particles);
    int total = 0;
    for (size_t i = 0; i < num_particles; ++i) {
        host_offsets[i] = total;
        total += host_counts[i];
    }
    num_pairs = total;

    CUDA_CHECK(cudaMemcpy(d_particle_offsets, host_offsets.data(), num_particles * sizeof(int), cudaMemcpyHostToDevice));
}

void NeighborList::get_host(std::vector<int>& host_list, std::vector<int>& host_counts) const {
    if (num_particles_allocated == 0) {
        host_list.clear();
        host_counts.clear();
        return;
    }

    if (d_neighbor_list == nullptr || d_neighbor_counts == nullptr) {
        throw std::runtime_error("Neighbor list has not been calculated on the GPU yet.");
    }

    host_list.resize(num_particles_allocated * max_neighbors);
    host_counts.resize(num_particles_allocated);

    CUDA_CHECK(cudaMemcpy(host_list.data(), d_neighbor_list, num_particles_allocated * max_neighbors * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_counts.data(), d_neighbor_counts, num_particles_allocated * sizeof(int), cudaMemcpyDeviceToHost));
}
