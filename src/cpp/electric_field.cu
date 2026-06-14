#include "electric_field.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>

// Macro to check CUDA errors and exit/throw if one occurs
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
      if (abort) exit(code);
   }
}

Electric_Field::Electric_Field(double box_x, double box_y, double box_z,
                               double errortol,
                               double xi,
                               bool calc_inter_dipole)
    : box_x(box_x), box_y(box_y), box_z(box_z), errortol(errortol),
      xi(xi), calc_inter_dipole(calc_inter_dipole),
      particles_updated(false), field_points_updated(false),
      dipoles_updated(false),
      num_particles(0), num_field_points(0),
      d_x_part(nullptr), d_y_part(nullptr), d_z_part(nullptr),
      d_x_field(nullptr), d_y_field(nullptr), d_z_field(nullptr),
      d_dipoles(nullptr),
      d_self_coef_r(nullptr), d_self_coef_i(nullptr),
      d_spread_coef(nullptr), d_spread_idxs(nullptr), num_spread(0),
      d_E_point(nullptr), d_particle_index(nullptr), d_contract_coef(nullptr), d_contract_idxs(nullptr), num_contract(0),
      d_self_perp(nullptr), d_perp(nullptr), d_para(nullptr), d_particle_offsets(nullptr), num_pairs(0),
      self_coef(0.0) {

    const double PI_CONST = 3.14159265358979323846;
    self_coef = -4.0 * std::pow(xi, 3.0) / (3.0 * std::sqrt(PI_CONST));

    // Run Ewald precalculations
    computePrecalculations();

    // Calculate real space tables and copy to GPU
    computeRealSpaceTables();

    // Allocate grid and FFT plan
    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    CUDA_CHECK(cudaMalloc(&d_fE_grid, grid_voxels * 3 * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fEs_grid, grid_voxels * 3 * 2 * sizeof(double)));

    int n[3] = { num_grid[0], num_grid[1], num_grid[2] };
    cufftResult plan_res = cufftPlanMany((cufftHandle*)&fft_plan, 3, n,
                                        n, 3, 1, // inembed, istride, idist
                                        n, 3, 1, // onembed, ostride, odist
                                        CUFFT_Z2Z, 3);
    if (plan_res != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT plan creation failed with code: " + std::to_string(plan_res));
    }
}

Electric_Field::~Electric_Field() {
    if (d_x_part) {
        cudaFree(d_x_part);
        d_x_part = nullptr;
    }
    if (d_y_part) {
        cudaFree(d_y_part);
        d_y_part = nullptr;
    }
    if (d_z_part) {
        cudaFree(d_z_part);
        d_z_part = nullptr;
    }
    if (d_dipoles) {
        cudaFree(d_dipoles);
        d_dipoles = nullptr;
    }
    if (d_self_coef_r) {
        cudaFree(d_self_coef_r);
        d_self_coef_r = nullptr;
    }
    if (d_self_coef_i) {
        cudaFree(d_self_coef_i);
        d_self_coef_i = nullptr;
    }
    if (d_spread_coef) {
        cudaFree(d_spread_coef);
        d_spread_coef = nullptr;
    }
    if (d_spread_idxs) {
        cudaFree(d_spread_idxs);
        d_spread_idxs = nullptr;
    }
    if (d_E_point) {
        cudaFree(d_E_point);
        d_E_point = nullptr;
    }
    if (d_particle_index) {
        cudaFree(d_particle_index);
        d_particle_index = nullptr;
    }
    if (d_contract_coef) {
        cudaFree(d_contract_coef);
        d_contract_coef = nullptr;
    }
    if (d_contract_idxs) {
        cudaFree(d_contract_idxs);
        d_contract_idxs = nullptr;
    }
    if (d_self_perp) {
        cudaFree(d_self_perp);
        d_self_perp = nullptr;
    }
    if (d_perp) {
        cudaFree(d_perp);
        d_perp = nullptr;
    }
    if (d_para) {
        cudaFree(d_para);
        d_para = nullptr;
    }
    if (d_particle_offsets) {
        cudaFree(d_particle_offsets);
        d_particle_offsets = nullptr;
    }
    if (!calc_inter_dipole) {
        if (d_x_field) {
            cudaFree(d_x_field);
            d_x_field = nullptr;
        }
        if (d_y_field) {
            cudaFree(d_y_field);
            d_y_field = nullptr;
        }
        if (d_z_field) {
            cudaFree(d_z_field);
            d_z_field = nullptr;
        }
    }
    if (d_neighbor_list) {
        cudaFree(d_neighbor_list);
        d_neighbor_list = nullptr;
    }
    if (d_neighbor_counts) {
        cudaFree(d_neighbor_counts);
        d_neighbor_counts = nullptr;
    }
    if (d_r_table) {
        cudaFree(d_r_table);
        d_r_table = nullptr;
    }
    if (d_field_dip_1) {
        cudaFree(d_field_dip_1);
        d_field_dip_1 = nullptr;
    }
    if (d_field_dip_2) {
        cudaFree(d_field_dip_2);
        d_field_dip_2 = nullptr;
    }
    if (d_offset) {
        cudaFree(d_offset);
        d_offset = nullptr;
    }
    if (d_offsetxyz) {
        cudaFree(d_offsetxyz);
        d_offsetxyz = nullptr;
    }
    if (d_scale_coef) {
        cudaFree(d_scale_coef);
        d_scale_coef = nullptr;
    }
    if (d_khat) {
        cudaFree(d_khat);
        d_khat = nullptr;
    }

    if (d_fE_grid) {
        cudaFree(d_fE_grid);
        d_fE_grid = nullptr;
    }
    if (d_fEs_grid) {
        cudaFree(d_fEs_grid);
        d_fEs_grid = nullptr;
    }
    if (fft_plan) {
        cufftDestroy((cufftHandle)fft_plan);
        fft_plan = 0;
    }
}

// CUDA kernel to compute neighbor list under periodic boundary conditions
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

void Electric_Field::computeNeighborList(int max_neighbors_per_particle) {
    if (num_particles == 0) return;

    // Check if we need to allocate or reallocate memory
    if (d_neighbor_list == nullptr || max_neighbors_per_particle != max_neighbors) {
        if (d_neighbor_list != nullptr) {
            CUDA_CHECK(cudaFree(d_neighbor_list));
        }
        max_neighbors = max_neighbors_per_particle;
        CUDA_CHECK(cudaMalloc(&d_neighbor_list, num_particles * max_neighbors * sizeof(int)));
    }

    if (d_neighbor_counts == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_neighbor_counts, num_particles * sizeof(int)));
    }

    // Initialize counts to 0
    CUDA_CHECK(cudaMemset(d_neighbor_counts, 0, num_particles * sizeof(int)));

    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    compute_neighbor_list_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_x_part, d_y_part, d_z_part,
        d_x_field, d_y_field, d_z_field,
        d_neighbor_list, d_neighbor_counts,
        num_particles,
        num_field_points,
        box_x, box_y, box_z,
        rc,
        max_neighbors,
        calc_inter_dipole
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Electric_Field::getNeighborListHost(std::vector<int>& host_list, std::vector<int>& host_counts) const {
    if (num_particles == 0) {
        host_list.clear();
        host_counts.clear();
        return;
    }

    if (d_neighbor_list == nullptr || d_neighbor_counts == nullptr) {
        throw std::runtime_error("Neighbor list has not been calculated on the GPU yet.");
    }

    host_list.resize(num_particles * max_neighbors);
    host_counts.resize(num_particles);

    CUDA_CHECK(cudaMemcpy(host_list.data(), d_neighbor_list, num_particles * max_neighbors * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_counts.data(), d_neighbor_counts, num_particles * sizeof(int), cudaMemcpyDeviceToHost));
}

void Electric_Field::computeRealSpaceTables() {
    size_t num_r_steps = 9000;
    std::vector<double> r_vals(num_r_steps);
    for (size_t idx = 0; idx < num_r_steps; ++idx) {
        r_vals[idx] = 1.0 + idx * 0.001;
    }

    const double PI = 3.14159265358979323846;
    double pi_pow_1_5 = std::pow(PI, 1.5);
    double xi_sq = xi * xi;
    double xi_cub = xi_sq * xi;
    double xi_4 = xi_sq * xi_sq;
    double xi_5 = xi_4 * xi;
    double xi_6 = xi_5 * xi;

    std::vector<double> fd1(num_r_steps);
    std::vector<double> fd2(num_r_steps);

    for (size_t idx = 0; idx < num_r_steps; ++idx) {
        double r = r_vals[idx];
        double r_sq = r * r;
        double r_cub = r_sq * r;
        double r_4 = r_sq * r_sq;
        double r_5 = r_4 * r;
        double r_6 = r_4 * r_sq;

        // --- field_dip_1 calculation ---
        double fd1_exppolyp = 1.0 / (1024.0 * pi_pow_1_5 * xi_5 * r_cub) * 
            (4.0 * xi_4 * r_5 - 8.0 * xi_4 * r_4 + 8.0 * xi_sq * (2.0 - 7.0 * xi_sq) * r_cub - 
             8.0 * xi_sq * (3.0 + 2.0 * xi_sq) * r_sq + (3.0 - 12.0 * xi_sq + 32.0 * xi_4) * r + 
             2.0 * (3.0 + 4.0 * xi_sq - 32.0 * xi_4));

        double fd1_exppolym = 1.0 / (1024.0 * pi_pow_1_5 * xi_5 * r_cub) * 
            (4.0 * xi_4 * r_5 + 8.0 * xi_4 * r_4 + 8.0 * xi_sq * (2.0 - 7.0 * xi_sq) * r_cub + 
             8.0 * xi_sq * (3.0 + 2.0 * xi_sq) * r_sq + (3.0 - 12.0 * xi_sq + 32.0 * xi_4) * r - 
             2.0 * (3.0 + 4.0 * xi_sq - 32.0 * xi_4));

        double fd1_exppoly0 = 1.0 / (512.0 * pi_pow_1_5 * xi_5 * r_sq) * 
            (-4.0 * xi_4 * r_4 - 8.0 * xi_sq * (2.0 - 9.0 * xi_sq) * r_sq - 3.0 + 36.0 * xi_sq);

        double fd1_erfpolyp = 1.0 / (2048.0 * PI * xi_6 * r_cub) * 
            (-8.0 * xi_6 * r_6 - 36.0 * xi_4 * (1.0 - 4.0 * xi_sq) * r_4 + 256.0 * xi_6 * r_cub - 
             18.0 * xi_sq * (1.0 - 8.0 * xi_sq) * r_sq + 3.0 - 36.0 * xi_sq + 256.0 * xi_6);

        double fd1_erfpolym = 1.0 / (2048.0 * PI * xi_6 * r_cub) * 
            (-8.0 * xi_6 * r_6 - 36.0 * xi_4 * (1.0 - 4.0 * xi_sq) * r_4 - 256.0 * xi_6 * r_cub - 
             18.0 * xi_sq * (1.0 - 8.0 * xi_sq) * r_sq + 3.0 - 36.0 * xi_sq + 256.0 * xi_6);

        double fd1_erfpoly0 = 1.0 / (1024.0 * PI * xi_6 * r_cub) * 
            (8.0 * xi_6 * r_6 + 36.0 * xi_4 * (1.0 - 4.0 * xi_sq) * r_4 + 
             18.0 * xi_sq * (1.0 - 8.0 * xi_sq) * r_sq - 3.0 + 36.0 * xi_sq);

        double fd1_reg = 0.0;
        if (calc_inter_dipole) {
            fd1_reg = -1.0 / (4.0 * PI * r_cub) + 1.0 / (4.0 * PI) * (1.0 - 9.0 * r / 16.0 + r_cub / 32.0);
        }

        double term_p = fd1_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + fd1_erfpolyp * std::erfc((r + 2.0) * xi);
        double term_m = fd1_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + fd1_erfpolym * std::erfc((r - 2.0) * xi);
        double term_0 = fd1_exppoly0 * std::exp(-r_sq * xi_sq) + fd1_erfpoly0 * std::erfc(r * xi);
        double reg_part = (r < 2.0) ? fd1_reg : 0.0;

        fd1[idx] = term_p + term_m + term_0 + reg_part;

        // --- field_dip_2 calculation ---
        double fd2_exppolyp = 1.0 / (512.0 * pi_pow_1_5 * xi_5 * r_cub) * 
            (8.0 * xi_4 * r_5 - 16.0 * xi_4 * r_4 + 2.0 * xi_sq * (7.0 - 20.0 * xi_sq) * r_cub - 
             4.0 * xi_sq * (3.0 - 4.0 * xi_sq) * r_sq - (3.0 - 12.0 * xi_sq + 32.0 * xi_4) * r - 
             2.0 * (3.0 + 4.0 * xi_sq - 32.0 * xi_4));

        double fd2_exppolym = 1.0 / (512.0 * pi_pow_1_5 * xi_5 * r_cub) * 
            (8.0 * xi_4 * r_5 + 16.0 * xi_4 * r_4 + 2.0 * xi_sq * (7.0 - 20.0 * xi_sq) * r_cub + 
             4.0 * xi_sq * (3.0 - 4.0 * xi_sq) * r_sq - (3.0 - 12.0 * xi_sq + 32.0 * xi_4) * r + 
             2.0 * (3.0 + 4.0 * xi_sq - 32.0 * xi_4));

        double fd2_exppoly0 = 1.0 / (256.0 * pi_pow_1_5 * xi_5 * r_sq) * 
            (-8.0 * xi_4 * r_4 - 2.0 * xi_sq * (7.0 - 36.0 * xi_sq) * r_sq + 3.0 - 36.0 * xi_sq);

        double fd2_erfpolyp = 1.0 / (1024.0 * PI * xi_6 * r_cub) * 
            (-16.0 * xi_6 * r_6 - 36.0 * xi_4 * (1.0 - 4.0 * xi_sq) * r_4 + 128.0 * xi_6 * r_cub - 3.0 + 36.0 * xi_sq - 256.0 * xi_6);

        double fd2_erfpolym = 1.0 / (1024.0 * PI * xi_6 * r_cub) * 
            (-16.0 * xi_6 * r_6 - 36.0 * xi_4 * (1.0 - 4.0 * xi_sq) * r_4 - 128.0 * xi_6 * r_cub - 3.0 + 36.0 * xi_sq - 256.0 * xi_6);

        double fd2_erfpoly0 = 1.0 / (512.0 * PI * xi_6 * r_cub) * 
            (16.0 * xi_6 * r_6 + 36.0 * xi_4 * (1.0 - 4.0 * xi_sq) * r_4 + 3.0 - 36.0 * xi_sq);

        double fd2_reg = 0.0;
        if (calc_inter_dipole) {
            fd2_reg = 1.0 / (2.0 * PI * r_cub) + 1.0 / (4.0 * PI) * (1.0 - 9.0 * r / 8.0 + r_cub / 8.0);
        }

        double fd2_term_p = fd2_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + fd2_erfpolyp * std::erfc((r + 2.0) * xi);
        double fd2_term_m = fd2_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + fd2_erfpolym * std::erfc((r - 2.0) * xi);
        double fd2_term_0 = fd2_exppoly0 * std::exp(-r_sq * xi_sq) + fd2_erfpoly0 * std::erfc(r * xi);
        double fd2_reg_part = (r < 2.0) ? fd2_reg : 0.0;

        fd2[idx] = fd2_term_p + fd2_term_m + fd2_term_0 + fd2_reg_part;
    }

    double selfo = (-1.0 + 6.0 * xi_sq + (1.0 - 2.0 * xi_sq) * std::exp(-4.0 * xi_sq)) / 
                   (16.0 * pi_pow_1_5 * xi_cub) + std::erfc(2.0 * xi) / (4.0 * PI);

    table_size = 9001;
    std::vector<double> host_r_table(table_size);
    std::vector<double> host_field_dip_1(table_size);
    std::vector<double> host_field_dip_2(table_size);

    host_r_table[0] = 0.0;
    host_field_dip_1[0] = selfo;
    host_field_dip_2[0] = selfo;

    for (size_t idx = 0; idx < num_r_steps; ++idx) {
        host_r_table[idx + 1] = r_vals[idx];
        host_field_dip_1[idx + 1] = fd1[idx];
        host_field_dip_2[idx + 1] = fd2[idx];
    }

    // Allocate GPU memory
    size_t size_in_bytes = table_size * sizeof(double);
    if (d_r_table) CUDA_CHECK(cudaFree(d_r_table));
    if (d_field_dip_1) CUDA_CHECK(cudaFree(d_field_dip_1));
    if (d_field_dip_2) CUDA_CHECK(cudaFree(d_field_dip_2));

    CUDA_CHECK(cudaMalloc(&d_r_table, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_field_dip_1, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_field_dip_2, size_in_bytes));

    // Copy host memory to device
    CUDA_CHECK(cudaMemcpy(d_r_table, host_r_table.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field_dip_1, host_field_dip_1.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field_dip_2, host_field_dip_2.data(), size_in_bytes, cudaMemcpyHostToDevice));
}

void Electric_Field::getRealSpaceTablesHost(std::vector<double>& host_r_table,
                                            std::vector<double>& host_field_dip_1,
                                            std::vector<double>& host_field_dip_2) const {
    if (table_size == 0 || d_r_table == nullptr || d_field_dip_1 == nullptr || d_field_dip_2 == nullptr) {
        throw std::runtime_error("Real space tables have not been calculated/allocated on the GPU yet.");
    }

    host_r_table.resize(table_size);
    host_field_dip_1.resize(table_size);
    host_field_dip_2.resize(table_size);

    size_t size_in_bytes = table_size * sizeof(double);
    CUDA_CHECK(cudaMemcpy(host_r_table.data(), d_r_table, size_in_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_field_dip_1.data(), d_field_dip_1, size_in_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_field_dip_2.data(), d_field_dip_2, size_in_bytes, cudaMemcpyDeviceToHost));
}

void Electric_Field::computePrecalculations() {
    // 1. Calculate rc
    rc = std::sqrt(-std::log(errortol)) / xi;

    // 2. Check if rc > box/2 for any dimension
    if (rc > box_x / 2.0 || rc > box_y / 2.0 || rc > box_z / 2.0) {
        throw std::runtime_error("Real space cutoff (" + std::to_string(rc) + ") larger than half the box length.");
    }

    // 3. Compute kcut
    double kcut = 2.0 * xi * xi * rc;

    const double PI = 3.14159265358979323846;

    // 4. Compute num_grid
    num_grid[0] = static_cast<int>(std::ceil(1.0 + box_x * kcut / PI));
    num_grid[1] = static_cast<int>(std::ceil(1.0 + box_y * kcut / PI));
    num_grid[2] = static_cast<int>(std::ceil(1.0 + box_z * kcut / PI));

    // 5. Compute grid_spacing
    grid_spacing[0] = box_x / num_grid[0];
    grid_spacing[1] = box_y / num_grid[1];
    grid_spacing[2] = box_z / num_grid[2];

    // 6. Compute spectral_split
    int num_grid_gaussian = static_cast<int>(std::ceil(-2.0 * std::log(errortol) / PI));
    spectral_split[0] = num_grid_gaussian * std::pow(grid_spacing[0] * xi, 2) / PI;
    spectral_split[1] = num_grid_gaussian * std::pow(grid_spacing[1] * xi, 2) / PI;
    spectral_split[2] = num_grid_gaussian * std::pow(grid_spacing[2] * xi, 2) / PI;

    // 7. Compute offsets
    int off = num_grid_gaussian / 2;
    int min_off = -off;
    int max_off = off + 1;

    std::vector<int> host_offset;
    std::vector<double> host_offsetxyz;

    for (int x_off = min_off; x_off < max_off; ++x_off) {
        for (int y_off = min_off; y_off < max_off; ++y_off) {
            for (int z_off = min_off; z_off < max_off; ++z_off) {
                // Store [z, y, x] (swapped columns)
                host_offset.push_back(z_off);
                host_offset.push_back(y_off);
                host_offset.push_back(x_off);

                // Store [z*spacing_x, y*spacing_y, x*spacing_z]
                // Note: column 0 (z) is multiplied by grid_spacing[0] (spacing_x)
                // Column 1 (y) is multiplied by grid_spacing[1] (spacing_y)
                // Column 2 (x) is multiplied by grid_spacing[2] (spacing_z)
                host_offsetxyz.push_back(z_off * grid_spacing[0]);
                host_offsetxyz.push_back(y_off * grid_spacing[1]);
                host_offsetxyz.push_back(x_off * grid_spacing[2]);
            }
        }
    }

    num_offsets = host_offset.size() / 3;

    // 8. Scale Precalcs (reciprocal space coefficients)
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

    // Origin indices
    int k0x = static_cast<int>(std::ceil((num_grid[0] - 1) / 2.0));
    int k0y = static_cast<int>(std::ceil((num_grid[1] - 1) / 2.0));
    int k0z = static_cast<int>(std::ceil((num_grid[2] - 1) / 2.0));

    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    std::vector<double> host_scale_coef(grid_voxels, 0.0);
    std::vector<double> host_khat(grid_voxels * 3, 0.0);

    for (int ix = 0; ix < num_grid[0]; ++ix) {
        for (int iy = 0; iy < num_grid[1]; ++iy) {
            for (int iz = 0; iz < num_grid[2]; ++iz) {
                size_t linear_idx = ix * num_grid[1] * num_grid[2] + iy * num_grid[2] + iz;

                if (ix == k0x && iy == k0y && iz == k0z) {
                    host_scale_coef[linear_idx] = 0.0;
                    host_khat[linear_idx * 3 + 0] = 0.0;
                    host_khat[linear_idx * 3 + 1] = 0.0;
                    host_khat[linear_idx * 3 + 2] = 0.0;
                    continue;
                }

                double kx_val = Kx[ix];
                double ky_val = Ky[iy];
                double kz_val = Kz[iz];

                double ksqx = kx_val * kx_val;
                double ksqy = ky_val * ky_val;
                double ksqz = kz_val * kz_val;

                double ksqsm = ksqx + ksqy + ksqz;
                double kmag = std::sqrt(ksqsm);

                host_khat[linear_idx * 3 + 0] = kx_val / kmag;
                host_khat[linear_idx * 3 + 1] = ky_val / kmag;
                host_khat[linear_idx * 3 + 2] = kz_val / kmag;

                double etaksq = ksqx * (1.0 - spectral_split[0]) + 
                                ksqy * (1.0 - spectral_split[1]) + 
                                ksqz * (1.0 - spectral_split[2]);

                // Analytical Spherical Bessel J_{3/2}(kmag) squared:
                // Bessel J_{3/2}(kmag)^2 = (2 / (pi * kmag)) * (sin(kmag)/kmag - cos(kmag))^2
                // scale_coef = (9 * pi) / (2 * kmag) * Bessel^2 * exp(-etaksq / (4 * xi^2)) / ksqsm
                //            = 9 * (sin(kmag)/kmag - cos(kmag))^2 * exp(...) / (ksqsm * ksqsm)
                double term = std::sin(kmag) / kmag - std::cos(kmag);
                double exp_part = std::exp(-etaksq / (4.0 * xi * xi));
                host_scale_coef[linear_idx] = 9.0 * term * term * exp_part / (ksqsm * ksqsm);
            }
        }
    }

    // Allocate device memory and copy
    if (d_offset) CUDA_CHECK(cudaFree(d_offset));
    if (d_offsetxyz) CUDA_CHECK(cudaFree(d_offsetxyz));
    if (d_scale_coef) CUDA_CHECK(cudaFree(d_scale_coef));
    if (d_khat) CUDA_CHECK(cudaFree(d_khat));

    CUDA_CHECK(cudaMalloc(&d_offset, num_offsets * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsetxyz, num_offsets * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scale_coef, grid_voxels * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_khat, grid_voxels * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_offset, host_offset.data(), num_offsets * 3 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsetxyz, host_offsetxyz.data(), num_offsets * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_coef, host_scale_coef.data(), grid_voxels * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_khat, host_khat.data(), grid_voxels * 3 * sizeof(double), cudaMemcpyHostToDevice));
}

void Electric_Field::getPrecalculationsHost(std::vector<int>& host_offset,
                                            std::vector<double>& host_offsetxyz,
                                            std::vector<double>& host_scale_coef,
                                            std::vector<double>& host_khat) const {
    if (num_offsets == 0 || d_offset == nullptr || d_offsetxyz == nullptr || 
        d_scale_coef == nullptr || d_khat == nullptr) {
        throw std::runtime_error("Ewald precalculations have not been calculated/allocated on GPU yet.");
    }

    host_offset.resize(num_offsets * 3);
    host_offsetxyz.resize(num_offsets * 3);
    
    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    host_scale_coef.resize(grid_voxels);
    host_khat.resize(grid_voxels * 3);

    CUDA_CHECK(cudaMemcpy(host_offset.data(), d_offset, num_offsets * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_offsetxyz.data(), d_offsetxyz, num_offsets * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_scale_coef.data(), d_scale_coef, grid_voxels * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_khat.data(), d_khat, grid_voxels * 3 * sizeof(double), cudaMemcpyDeviceToHost));
}

void Electric_Field::updateParticleCoordinates(const std::vector<double>& x_part,
                                               const std::vector<double>& y_part,
                                               const std::vector<double>& z_part) {
    if (x_part.size() != y_part.size() || x_part.size() != z_part.size()) {
        throw std::invalid_argument("Input coordinate vectors must have the exact same size.");
    }
    size_t new_num_particles = x_part.size();
    if (new_num_particles == 0) return;

    if (new_num_particles != num_particles) {
        if (d_x_part) cudaFree(d_x_part);
        if (d_y_part) cudaFree(d_y_part);
        if (d_z_part) cudaFree(d_z_part);
        if (d_dipoles) cudaFree(d_dipoles);
        if (d_self_coef_r) cudaFree(d_self_coef_r);
        if (d_self_coef_i) cudaFree(d_self_coef_i);

        num_particles = new_num_particles;
        size_t size_part_bytes = num_particles * sizeof(double);
        size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_x_part, size_part_bytes));
        CUDA_CHECK(cudaMalloc(&d_y_part, size_part_bytes));
        CUDA_CHECK(cudaMalloc(&d_z_part, size_part_bytes));
        CUDA_CHECK(cudaMalloc(&d_dipoles, size_dipoles_bytes));
        CUDA_CHECK(cudaMalloc(&d_self_coef_r, size_part_bytes));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, size_part_bytes));

        CUDA_CHECK(cudaMemset(d_dipoles, 0, size_dipoles_bytes));
    }

    std::vector<double> host_sc_r(num_particles, self_coef);
    std::vector<double> host_sc_i(num_particles, 0.0);
    CUDA_CHECK(cudaMemcpy(d_self_coef_r, host_sc_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, host_sc_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));

    size_t size_part_bytes = num_particles * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_x_part, x_part.data(), size_part_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_part, y_part.data(), size_part_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_part, z_part.data(), size_part_bytes, cudaMemcpyHostToDevice));

    particles_updated = true;

    if (calc_inter_dipole) {
        d_x_field = d_x_part;
        d_y_field = d_y_part;
        d_z_field = d_z_part;
        num_field_points = num_particles;
        field_points_updated = true;
    }
}

void Electric_Field::updateFieldCoordinates(const std::vector<double>& x_field,
                                            const std::vector<double>& y_field,
                                            const std::vector<double>& z_field) {
    if (calc_inter_dipole) {
        updateParticleCoordinates(x_field, y_field, z_field);
        return;
    }

    if (x_field.size() != y_field.size() || x_field.size() != z_field.size()) {
        throw std::invalid_argument("Input coordinate vectors must have the exact same size.");
    }
    size_t new_num_field = x_field.size();
    if (new_num_field == 0) return;

    if (new_num_field != num_field_points) {
        if (d_x_field) cudaFree(d_x_field);
        if (d_y_field) cudaFree(d_y_field);
        if (d_z_field) cudaFree(d_z_field);

        num_field_points = new_num_field;
        size_t size_field_bytes = num_field_points * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_x_field, size_field_bytes));
        CUDA_CHECK(cudaMalloc(&d_y_field, size_field_bytes));
        CUDA_CHECK(cudaMalloc(&d_z_field, size_field_bytes));
    }

    size_t size_field_bytes = num_field_points * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_x_field, x_field.data(), size_field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_field, y_field.data(), size_field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_field, z_field.data(), size_field_bytes, cudaMemcpyHostToDevice));

    field_points_updated = true;
}

void Electric_Field::updateDipoles(const std::vector<double>& dip_x,
                                   const std::vector<double>& dip_y,
                                   const std::vector<double>& dip_z) {
    if (dip_x.size() != num_particles || dip_y.size() != num_particles || dip_z.size() != num_particles) {
        throw std::invalid_argument("Input dipole vectors must match the allocated number of particles.");
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

void Electric_Field::updateDipolesComplex(const std::vector<double>& dip_xr, const std::vector<double>& dip_xi,
                                          const std::vector<double>& dip_yr, const std::vector<double>& dip_yi,
                                          const std::vector<double>& dip_zr, const std::vector<double>& dip_zi) {
    if (dip_xr.size() != num_particles || dip_xi.size() != num_particles ||
        dip_yr.size() != num_particles || dip_yi.size() != num_particles ||
        dip_zr.size() != num_particles || dip_zi.size() != num_particles) {
        throw std::invalid_argument("Input complex dipole component vectors must match the allocated number of particles.");
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

void Electric_Field::getDipolesHost(std::vector<double>& host_dip_x,
                                    std::vector<double>& host_dip_y,
                                    std::vector<double>& host_dip_z) const {
    if (num_particles == 0 || d_dipoles == nullptr) {
        throw std::runtime_error("Dipoles have not been allocated on GPU yet.");
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

void Electric_Field::getDipolesComplexHost(std::vector<double>& host_dip_xr, std::vector<double>& host_dip_xi,
                                           std::vector<double>& host_dip_yr, std::vector<double>& host_dip_yi,
                                           std::vector<double>& host_dip_zr, std::vector<double>& host_dip_zi) const {
    if (num_particles == 0 || d_dipoles == nullptr) {
        throw std::runtime_error("Dipoles have not been allocated on GPU yet.");
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

void Electric_Field::setSelfCoef(const std::vector<double>& self_coef_r, const std::vector<double>& self_coef_i) {
    if (self_coef_r.size() != num_particles || self_coef_i.size() != num_particles) {
        throw std::invalid_argument("Self coefficient vectors must match the number of particles.");
    }
    if (num_particles == 0) return;

    CUDA_CHECK(cudaMemcpy(d_self_coef_r, self_coef_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, self_coef_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
}

void Electric_Field::setSelfCoef(double val_r, double val_i) {
    if (num_particles == 0) return;
    std::vector<double> host_sc_r(num_particles, val_r);
    std::vector<double> host_sc_i(num_particles, val_i);
    CUDA_CHECK(cudaMemcpy(d_self_coef_r, host_sc_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, host_sc_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
}

__device__ static double interpolate_table_gpu(double r, const double* r_table, const double* y_table, size_t table_size) {
    if (r <= 0.0) return y_table[0];
    if (r >= r_table[table_size - 1]) return y_table[table_size - 1];

    if (r < 1.0) {
        double t = r;
        return y_table[0] * (1.0 - t) + y_table[1] * t;
    }

    size_t idx = 1 + static_cast<size_t>((r - 1.0) / 0.001);
    if (idx >= table_size - 1) {
        return y_table[table_size - 1];
    }
    double r0 = r_table[idx];
    double r1 = r_table[idx + 1];
    double t = (r - r0) / (r1 - r0);
    return y_table[idx] * (1.0 - t) + y_table[idx + 1] * t;
}

__global__ void spread_precalcs_kernel(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
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
    double spectral_split_x,
    double spectral_split_y,
    double spectral_split_z,
    double const_factor,
    double xi)
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

    int ox = offset[o * 3 + 0];
    int oy = offset[o * 3 + 1];
    int oz = offset[o * 3 + 2];

    double oxyz_x = offsetxyz[o * 3 + 0];
    double oxyz_y = offsetxyz[o * 3 + 1];
    double oxyz_z = offsetxyz[o * 3 + 2];

    int geix = ((gix + ox - 1) % num_grid_x + num_grid_x) % num_grid_x;
    int geiy = ((giy + oy - 1) % num_grid_y + num_grid_y) % num_grid_y;
    int geiz = ((giz + oz - 1) % num_grid_z + num_grid_z) % num_grid_z;

    spread_idxs[idx * 3 + 0] = geix;
    spread_idxs[idx * 3 + 1] = geiy;
    spread_idxs[idx * 3 + 2] = geiz;

    double gdx = dx + oxyz_x;
    double gdy = dy + oxyz_y;
    double gdz = dz + oxyz_z;

    double div_eta = (gdx * gdx / spectral_split_x) +
                     (gdy * gdy / spectral_split_y) +
                     (gdz * gdz / spectral_split_z);

    spread_coef[idx] = const_factor * exp(-2.0 * xi * xi * div_eta);
}

__global__ void contract_precalcs_kernel(
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
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
    double spectral_split_x,
    double spectral_split_y,
    double spectral_split_z,
    double const_factor,
    double xi)
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

    int ox = offset[o * 3 + 0];
    int oy = offset[o * 3 + 1];
    int oz = offset[o * 3 + 2];

    double oxyz_x = offsetxyz[o * 3 + 0];
    double oxyz_y = offsetxyz[o * 3 + 1];
    double oxyz_z = offsetxyz[o * 3 + 2];

    int geix = ((gix + ox - 1) % num_grid_x + num_grid_x) % num_grid_x;
    int geiy = ((giy + oy - 1) % num_grid_y + num_grid_y) % num_grid_y;
    int geiz = ((giz + oz - 1) % num_grid_z + num_grid_z) % num_grid_z;

    contract_idxs[idx * 3 + 0] = geix;
    contract_idxs[idx * 3 + 1] = geiy;
    contract_idxs[idx * 3 + 2] = geiz;

    particle_index[idx] = i;

    double gdx = dx + oxyz_x;
    double gdy = dy + oxyz_y;
    double gdz = dz + oxyz_z;

    double div_eta = (gdx * gdx / spectral_split_x) +
                     (gdy * gdy / spectral_split_y) +
                     (gdz * gdz / spectral_split_z);

    contract_coef[idx] = const_factor * exp(-2.0 * xi * xi * div_eta);
}

__global__ void real_space_precalcs_kernel(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
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
            double p_val = interpolate_table_gpu(d, r_table, field_dip_1, table_size);
            double a_val = interpolate_table_gpu(d, r_table, field_dip_2, table_size);

            perp[start_idx + k] = p_val;
            para[start_idx + k] = a_val;
        }
    }
}


void Electric_Field::spreadPrecalcs() {
    if (num_particles == 0) return;

    num_spread = num_particles * num_offsets;
    size_t size_coef_bytes = num_spread * sizeof(double);
    size_t size_idxs_bytes = num_spread * 3 * sizeof(int);

    if (d_spread_coef) CUDA_CHECK(cudaFree(d_spread_coef));
    if (d_spread_idxs) CUDA_CHECK(cudaFree(d_spread_idxs));

    CUDA_CHECK(cudaMalloc(&d_spread_coef, size_coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_spread_idxs, size_idxs_bytes));

    const double PI = 3.14159265358979323846;
    double prod_split = spectral_split[0] * spectral_split[1] * spectral_split[2];
    double const_factor = std::pow(2.0 * xi * xi / PI, 1.5) * std::sqrt(1.0 / prod_split);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_spread + threadsPerBlock - 1) / threadsPerBlock;

    spread_precalcs_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_x_part, d_y_part, d_z_part,
        d_offset, d_offsetxyz,
        d_spread_coef, d_spread_idxs,
        num_particles, num_offsets,
        grid_spacing[0], grid_spacing[1], grid_spacing[2],
        num_grid[0], num_grid[1], num_grid[2],
        spectral_split[0], spectral_split[1], spectral_split[2],
        const_factor, xi
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Electric_Field::contractPrecalcs() {
    if (num_field_points == 0) return;

    num_contract = num_field_points * num_offsets;
    size_t size_coef_bytes = num_contract * sizeof(double);
    size_t size_idxs_bytes = num_contract * 3 * sizeof(int);
    size_t size_part_idx_bytes = num_contract * sizeof(int);
    size_t size_epoint_bytes = num_field_points * 3 * 2 * sizeof(double);

    if (d_E_point) CUDA_CHECK(cudaFree(d_E_point));
    if (d_particle_index) CUDA_CHECK(cudaFree(d_particle_index));
    if (d_contract_coef) CUDA_CHECK(cudaFree(d_contract_coef));
    if (d_contract_idxs) CUDA_CHECK(cudaFree(d_contract_idxs));

    CUDA_CHECK(cudaMalloc(&d_E_point, size_epoint_bytes));
    CUDA_CHECK(cudaMemset(d_E_point, 0, size_epoint_bytes));

    CUDA_CHECK(cudaMalloc(&d_particle_index, size_part_idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_contract_coef, size_coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_contract_idxs, size_idxs_bytes));

    const double PI = 3.14159265358979323846;
    double prod_split = spectral_split[0] * spectral_split[1] * spectral_split[2];
    double prod_spacing = grid_spacing[0] * grid_spacing[1] * grid_spacing[2];
    double const_factor = std::pow(2.0 * xi * xi / PI, 1.5) * std::sqrt(1.0 / prod_split) * prod_spacing;

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_contract + threadsPerBlock - 1) / threadsPerBlock;

    contract_precalcs_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_x_field, d_y_field, d_z_field,
        d_offset, d_offsetxyz,
        d_contract_coef, d_contract_idxs, d_particle_index,
        num_field_points, num_offsets,
        grid_spacing[0], grid_spacing[1], grid_spacing[2],
        num_grid[0], num_grid[1], num_grid[2],
        spectral_split[0], spectral_split[1], spectral_split[2],
        const_factor, xi
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Electric_Field::realSpacePrecalcs() {
    if (d_neighbor_list == nullptr) {
        computeNeighborList(128);
    }

    std::vector<int> host_counts(num_particles);
    CUDA_CHECK(cudaMemcpy(host_counts.data(), d_neighbor_counts, num_particles * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> host_offsets(num_particles);
    int total = 0;
    for (size_t i = 0; i < num_particles; ++i) {
        host_offsets[i] = total;
        total += host_counts[i];
    }
    num_pairs = total;

    if (d_particle_offsets) CUDA_CHECK(cudaFree(d_particle_offsets));
    CUDA_CHECK(cudaMalloc(&d_particle_offsets, num_particles * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_particle_offsets, host_offsets.data(), num_particles * sizeof(int), cudaMemcpyHostToDevice));

    if (d_self_perp) CUDA_CHECK(cudaFree(d_self_perp));
    if (d_perp) CUDA_CHECK(cudaFree(d_perp));
    if (d_para) CUDA_CHECK(cudaFree(d_para));

    CUDA_CHECK(cudaMalloc(&d_self_perp, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_self_perp, d_field_dip_1, sizeof(double), cudaMemcpyDeviceToDevice));

    if (num_pairs > 0) {
        CUDA_CHECK(cudaMalloc(&d_perp, num_pairs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_para, num_pairs * sizeof(double)));

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

        real_space_precalcs_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_x_part, d_y_part, d_z_part,
            d_x_field, d_y_field, d_z_field,
            d_neighbor_list, d_neighbor_counts, d_particle_offsets,
            d_r_table, d_field_dip_1, d_field_dip_2,
            table_size,
            d_perp, d_para,
            num_particles,
            max_neighbors,
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

void Electric_Field::getSpreadPrecalcsHost(std::vector<double>& host_spread_coef,
                                           std::vector<int>& host_spread_idxs) const {
    if (num_spread == 0 || d_spread_coef == nullptr || d_spread_idxs == nullptr) {
        throw std::runtime_error("Spread precalcs have not been allocated on GPU yet.");
    }

    host_spread_coef.resize(num_spread);
    host_spread_idxs.resize(num_spread * 3);

    CUDA_CHECK(cudaMemcpy(host_spread_coef.data(), d_spread_coef, num_spread * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_spread_idxs.data(), d_spread_idxs, num_spread * 3 * sizeof(int), cudaMemcpyDeviceToHost));
}

void Electric_Field::getContractPrecalcsHost(std::vector<double>& host_E_point,
                                             std::vector<int>& host_particle_index,
                                             std::vector<double>& host_contract_coef,
                                             std::vector<int>& host_contract_idxs) const {
    if (num_contract == 0 || d_E_point == nullptr || d_particle_index == nullptr ||
        d_contract_coef == nullptr || d_contract_idxs == nullptr) {
        throw std::runtime_error("Contract precalcs have not been allocated on GPU yet.");
    }

    host_E_point.resize(num_field_points * 3 * 2);
    host_particle_index.resize(num_contract);
    host_contract_coef.resize(num_contract);
    host_contract_idxs.resize(num_contract * 3);

    CUDA_CHECK(cudaMemcpy(host_E_point.data(), d_E_point, num_field_points * 3 * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_particle_index.data(), d_particle_index, num_contract * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_contract_coef.data(), d_contract_coef, num_contract * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_contract_idxs.data(), d_contract_idxs, num_contract * 3 * sizeof(int), cudaMemcpyDeviceToHost));
}

void Electric_Field::getRealSpacePrecalcsHost(double& host_self_perp,
                                              std::vector<double>& host_perp,
                                              std::vector<double>& host_para) const {
    if (d_self_perp == nullptr) {
        throw std::runtime_error("Real space precalcs have not been allocated on GPU yet.");
    }

    CUDA_CHECK(cudaMemcpy(&host_self_perp, d_self_perp, sizeof(double), cudaMemcpyDeviceToHost));

    host_perp.resize(num_pairs);
    host_para.resize(num_pairs);

    if (num_pairs > 0) {
        if (d_perp == nullptr || d_para == nullptr) {
            throw std::runtime_error("Real space precalcs: perp or para is null but num_pairs > 0.");
        }
        CUDA_CHECK(cudaMemcpy(host_perp.data(), d_perp, num_pairs * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_para.data(), d_para, num_pairs * sizeof(double), cudaMemcpyDeviceToHost));
    }
}

__global__ void spread_kernel(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_spread) return;

    int i = idx / num_offsets;

    double dx_r = d_dipoles[(i * 3 + 0) * 2 + 0];
    double dx_i = d_dipoles[(i * 3 + 0) * 2 + 1];
    double dy_r = d_dipoles[(i * 3 + 1) * 2 + 0];
    double dy_i = d_dipoles[(i * 3 + 1) * 2 + 1];
    double dz_r = d_dipoles[(i * 3 + 2) * 2 + 0];
    double dz_i = d_dipoles[(i * 3 + 2) * 2 + 1];

    double coef = spread_coef[idx];

    int gx = spread_idxs[idx * 3 + 0];
    int gy = spread_idxs[idx * 3 + 1];
    int gz = spread_idxs[idx * 3 + 2];

    size_t grid_linear_idx = (static_cast<size_t>(gx) * num_grid_y * num_grid_z +
                              static_cast<size_t>(gy) * num_grid_z +
                              static_cast<size_t>(gz)) * 3;

    atomicAdd(&fE_grid[(grid_linear_idx + 0) * 2 + 0], coef * dx_r);
    atomicAdd(&fE_grid[(grid_linear_idx + 0) * 2 + 1], coef * dx_i);
    atomicAdd(&fE_grid[(grid_linear_idx + 1) * 2 + 0], coef * dy_r);
    atomicAdd(&fE_grid[(grid_linear_idx + 1) * 2 + 1], coef * dy_i);
    atomicAdd(&fE_grid[(grid_linear_idx + 2) * 2 + 0], coef * dz_r);
    atomicAdd(&fE_grid[(grid_linear_idx + 2) * 2 + 1], coef * dz_i);
}

__global__ void scale_kernel(
    double* __restrict__ fE_grid,
    const double* __restrict__ scale_coef,
    const double* __restrict__ khat,
    size_t num_voxels)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;

    double kx = khat[v * 3 + 0];
    double ky = khat[v * 3 + 1];
    double kz = khat[v * 3 + 2];

    double sc = scale_coef[v];

    double fr0_r = fE_grid[(v * 3 + 0) * 2 + 0];
    double fr1_r = fE_grid[(v * 3 + 1) * 2 + 0];
    double fr2_r = fE_grid[(v * 3 + 2) * 2 + 0];

    double fr0_i = fE_grid[(v * 3 + 0) * 2 + 1];
    double fr1_i = fE_grid[(v * 3 + 1) * 2 + 1];
    double fr2_i = fE_grid[(v * 3 + 2) * 2 + 1];

    double dot_r = fr0_r * kx + fr1_r * ky + fr2_r * kz;
    double dot_i = fr0_i * kx + fr1_i * ky + fr2_i * kz;

    double sum_r = sc * dot_r;
    double sum_i = sc * dot_i;

    fE_grid[(v * 3 + 0) * 2 + 0] = kx * sum_r;
    fE_grid[(v * 3 + 0) * 2 + 1] = kx * sum_i;

    fE_grid[(v * 3 + 1) * 2 + 0] = ky * sum_r;
    fE_grid[(v * 3 + 1) * 2 + 1] = ky * sum_i;

    fE_grid[(v * 3 + 2) * 2 + 0] = kz * sum_r;
    fE_grid[(v * 3 + 2) * 2 + 1] = kz * sum_i;
}

__global__ void contract_kernel(
    const double* __restrict__ Es_grid,
    const int* __restrict__ contract_idxs,
    const int* __restrict__ particle_index,
    const double* __restrict__ contract_coef,
    double* __restrict__ E_point,
    size_t num_contract,
    int num_grid_y,
    int num_grid_z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contract) return;

    int p = particle_index[idx];

    int gx = contract_idxs[idx * 3 + 0];
    int gy = contract_idxs[idx * 3 + 1];
    int gz = contract_idxs[idx * 3 + 2];

    size_t v = static_cast<size_t>(gx) * num_grid_y * num_grid_z +
               static_cast<size_t>(gy) * num_grid_z +
               static_cast<size_t>(gz);

    double coef = contract_coef[idx];

    for (int c = 0; c < 3; ++c) {
        double grid_r = Es_grid[(v * 3 + c) * 2 + 0];
        double grid_i = Es_grid[(v * 3 + c) * 2 + 1];

        atomicAdd(&E_point[(p * 3 + c) * 2 + 0], coef * grid_r);
        atomicAdd(&E_point[(p * 3 + c) * 2 + 1], coef * grid_i);
    }
}

__global__ void real_space_self_kernel(
    const double* __restrict__ d_dipoles,
    const double* __restrict__ d_self_coef_r,
    const double* __restrict__ d_self_coef_i,
    double self_perp,
    double* __restrict__ E_point,
    size_t num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    double sc_r = d_self_coef_r[i];
    double sc_i = d_self_coef_i[i];

    double factor_r = sc_r + self_perp;
    double factor_i = sc_i;

    double dx_r = d_dipoles[(i * 3 + 0) * 2 + 0];
    double dx_i = d_dipoles[(i * 3 + 0) * 2 + 1];
    double dy_r = d_dipoles[(i * 3 + 1) * 2 + 0];
    double dy_i = d_dipoles[(i * 3 + 1) * 2 + 1];
    double dz_r = d_dipoles[(i * 3 + 2) * 2 + 0];
    double dz_i = d_dipoles[(i * 3 + 2) * 2 + 1];

    atomicAdd(&E_point[(i * 3 + 0) * 2 + 0], factor_r * dx_r - factor_i * dx_i);
    atomicAdd(&E_point[(i * 3 + 0) * 2 + 1], factor_r * dx_i + factor_i * dx_r);

    atomicAdd(&E_point[(i * 3 + 1) * 2 + 0], factor_r * dy_r - factor_i * dy_i);
    atomicAdd(&E_point[(i * 3 + 1) * 2 + 1], factor_r * dy_i + factor_i * dy_r);

    atomicAdd(&E_point[(i * 3 + 2) * 2 + 0], factor_r * dz_r - factor_i * dz_i);
    atomicAdd(&E_point[(i * 3 + 2) * 2 + 1], factor_r * dz_i + factor_i * dz_r);
}

__global__ void real_space_neighbor_kernel(
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
    int start_idx = particle_offsets[i];

    double xi = x_part[i];
    double yi = y_part[i];
    double zi = z_part[i];

    double dip_x_r = d_dipoles[(i * 3 + 0) * 2 + 0];
    double dip_x_i = d_dipoles[(i * 3 + 0) * 2 + 1];
    double dip_y_r = d_dipoles[(i * 3 + 1) * 2 + 0];
    double dip_y_i = d_dipoles[(i * 3 + 1) * 2 + 1];
    double dip_z_r = d_dipoles[(i * 3 + 2) * 2 + 0];
    double dip_z_i = d_dipoles[(i * 3 + 2) * 2 + 1];

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

            atomicAdd(&E_point[(j * 3 + 0) * 2 + 0], contrib_x_r);
            atomicAdd(&E_point[(j * 3 + 0) * 2 + 1], contrib_x_i);

            atomicAdd(&E_point[(j * 3 + 1) * 2 + 0], contrib_y_r);
            atomicAdd(&E_point[(j * 3 + 1) * 2 + 1], contrib_y_i);

            atomicAdd(&E_point[(j * 3 + 2) * 2 + 0], contrib_z_r);
            atomicAdd(&E_point[(j * 3 + 2) * 2 + 1], contrib_z_i);
        }
    }
}

void Electric_Field::spread(double* d_fE_grid) {
    if (num_spread == 0 || d_spread_coef == nullptr || d_spread_idxs == nullptr || d_fE_grid == nullptr) {
        throw std::runtime_error("spread: Buffers/Precalcs are not allocated.");
    }

    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    CUDA_CHECK(cudaMemset(d_fE_grid, 0, grid_voxels * 3 * 2 * sizeof(double)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_spread + threadsPerBlock - 1) / threadsPerBlock;

    spread_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_dipoles,
        d_spread_coef, d_spread_idxs,
        d_fE_grid,
        num_spread, num_offsets,
        num_grid[0], num_grid[1], num_grid[2]
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Electric_Field::scale(double* d_fE_grid) {
    size_t grid_voxels = num_grid[0] * num_grid[1] * num_grid[2];
    if (grid_voxels == 0 || d_scale_coef == nullptr || d_khat == nullptr || d_fE_grid == nullptr) {
        throw std::runtime_error("scale: Buffers/Precalcs are not allocated.");
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (grid_voxels + threadsPerBlock - 1) / threadsPerBlock;

    scale_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_fE_grid,
        d_scale_coef,
        d_khat,
        grid_voxels
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Electric_Field::contract(double* d_E_point, const double* d_Es_grid) {
    if (num_contract == 0 || d_particle_index == nullptr || d_contract_coef == nullptr ||
        d_contract_idxs == nullptr || d_E_point == nullptr || d_Es_grid == nullptr) {
        throw std::runtime_error("contract: Buffers/Precalcs are not allocated.");
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_contract + threadsPerBlock - 1) / threadsPerBlock;

    contract_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_Es_grid,
        d_contract_idxs,
        d_particle_index,
        d_contract_coef,
        d_E_point,
        num_contract,
        num_grid[1],
        num_grid[2]
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Electric_Field::realSpace(double* d_E_point) {
    if (num_particles == 0 || d_E_point == nullptr) return;

    if (calc_inter_dipole) {
        if (d_self_perp == nullptr) {
            throw std::runtime_error("realSpace: d_self_perp is not calculated.");
        }
        double host_self_perp = 0.0;
        CUDA_CHECK(cudaMemcpy(&host_self_perp, d_self_perp, sizeof(double), cudaMemcpyDeviceToHost));

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

        real_space_self_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_dipoles, d_self_coef_r, d_self_coef_i,
            host_self_perp,
            d_E_point,
            num_particles
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (num_pairs > 0) {
        if (d_perp == nullptr || d_para == nullptr || d_particle_offsets == nullptr) {
            throw std::runtime_error("realSpace: perp/para precalcs are not allocated.");
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

        real_space_neighbor_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_x_part, d_y_part, d_z_part,
            d_x_field, d_y_field, d_z_field,
            d_dipoles,
            d_neighbor_list, d_neighbor_counts, d_particle_offsets,
            d_perp, d_para,
            d_E_point,
            num_particles,
            max_neighbors,
            box_x, box_y, box_z,
            rc
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}



__global__ void fftshift_3d_kernel(
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

    int xs = (x + N0 / 2) % N0;
    int ys = (y + N1 / 2) % N1;
    int zs = (z + N2 / 2) % N2;

    size_t shifted_idx = (static_cast<size_t>(xs) * N1 * N2 +
                          static_cast<size_t>(ys) * N2 +
                          static_cast<size_t>(zs));

    for (int c = 0; c < 3; ++c) {
        output[(shifted_idx * 3 + c) * 2 + 0] = input[(idx * 3 + c) * 2 + 0];
        output[(shifted_idx * 3 + c) * 2 + 1] = input[(idx * 3 + c) * 2 + 1];
    }
}

__global__ void ifftshift_3d_kernel(
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

    size_t shifted_idx = (static_cast<size_t>(xs) * N1 * N2 +
                          static_cast<size_t>(ys) * N2 +
                          static_cast<size_t>(zs));

    double scale_factor = 1.0 / (static_cast<double>(N0) * N1 * N2);

    for (int c = 0; c < 3; ++c) {
        output[(shifted_idx * 3 + c) * 2 + 0] = input[(idx * 3 + c) * 2 + 0] * scale_factor;
        output[(shifted_idx * 3 + c) * 2 + 1] = input[(idx * 3 + c) * 2 + 1] * scale_factor;
    }
}

void Electric_Field::electricField() {
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
    fftshift_3d_kernel<<<blocksPerGridShift, threadsPerBlock>>>(
        d_fE_grid, d_fEs_grid, num_grid[0], num_grid[1], num_grid[2]
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    scale(d_fEs_grid);
    
    ifftshift_3d_kernel<<<blocksPerGridShift, threadsPerBlock>>>(
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

void Electric_Field::calculate() {
    if (particles_updated || field_points_updated || d_self_perp == nullptr) {
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
