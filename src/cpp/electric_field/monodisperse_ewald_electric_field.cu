#include "monodisperse_ewald_electric_field.h"
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









Monodisperse_Ewald_Electric_Field::Monodisperse_Ewald_Electric_Field(double box_x, double box_y, double box_z,
                           double errortol,
                           double xi,
                           FieldCalcMode mode,
                           double radius,
                           bool solve_quadrupoles,
                           const std::vector<int>& quad_idxs,
                           PrecisionMode recip_precision)
    : Ewald_Electric_Field_Base(box_x, box_y, box_z, errortol, xi, mode, solve_quadrupoles, quad_idxs, recip_precision) {

    const double PI_CONST = 3.14159265358979323846;
    self_coef = -4.0 * std::pow(xi, 3.0) / (3.0 * std::sqrt(PI_CONST));

    // Run Ewald precalculations
    computePrecalculations();

    // Calculate real space tables and copy to GPU
    computeRealSpaceTables();

    // Allocate grid and FFT plan
    size_t grid_voxels = static_cast<size_t>(num_grid[0]) * num_grid[1] * num_grid[2];
    size_t element_size = use_recip_fp32 ? sizeof(float) : sizeof(double);
    cufftType fft_type = use_recip_fp32 ? CUFFT_C2C : CUFFT_Z2Z;

    CUDA_CHECK(cudaMalloc(&d_fE_grid, grid_voxels * 3 * 2 * element_size));

    int n[3] = { num_grid[0], num_grid[1], num_grid[2] };
    cufftResult plan_res = cufftPlanMany((cufftHandle*)&fft_plan, 3, n,
                                        n, 3, 1, // inembed, istride, idist
                                        n, 3, 1, // onembed, ostride, odist
                                        fft_type, 3);
    if (plan_res != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT plan creation failed with code: " + std::to_string(plan_res));
    }

    if (solve_quadrupoles) {
        CUDA_CHECK(cudaMalloc(&d_fG_grid, grid_voxels * 5 * 2 * element_size));

        cufftResult plan_res_G = cufftPlanMany((cufftHandle*)&fft_plan_G, 3, n,
                                              n, 5, 1, // inembed, istride, idist
                                              n, 5, 1, // onembed, ostride, odist
                                              fft_type, 5);
        if (plan_res_G != CUFFT_SUCCESS) {
            throw std::runtime_error("cuFFT plan G creation failed with code: " + std::to_string(plan_res_G));
        }
    }
}

Monodisperse_Ewald_Electric_Field::~Monodisperse_Ewald_Electric_Field() {
    if (d_self_perp) {
        cudaFree(d_self_perp);
        d_self_perp = nullptr;
    }
    if (d_khat) {
        cudaFree(d_khat);
        d_khat = nullptr;
    }
}

void Monodisperse_Ewald_Electric_Field::updateParticleCoordinates(const std::vector<double>& x_part,
                                             const std::vector<double>& y_part,
                                             const std::vector<double>& z_part) {
    Ewald_Electric_Field_Base::updateParticleCoordinates(x_part, y_part, z_part);
}
void Monodisperse_Ewald_Electric_Field::computeRealSpaceTables() {
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
    std::vector<double> fq1(num_r_steps);
    std::vector<double> fq2(num_r_steps);
    std::vector<double> fq3(num_r_steps);
    std::vector<double> gq1(num_r_steps);
    std::vector<double> gq2(num_r_steps);
    std::vector<double> gq3(num_r_steps);
    std::vector<double> gq4(num_r_steps);

    double xi_7 = xi_6 * xi;
    double xi_8 = xi_7 * xi;
    double xi_9 = xi_8 * xi;
    double xi_10 = xi_9 * xi;

    for (size_t idx = 0; idx < num_r_steps; ++idx) {
        double r = r_vals[idx];
        double r_sq = r * r;
        double r_cub = r_sq * r;
        double r_4 = r_sq * r_sq;
        double r_5 = r_4 * r;
        double r_6 = r_4 * r_sq;
        double r_8 = r_4 * r_4;
        double r_10 = r_5 * r_5;

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

        double fd1_reg = -1.0 / (4.0 * PI * r_cub) + 1.0 / (4.0 * PI) * (1.0 - 9.0 * r / 16.0 + r_cub / 32.0);

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

        double fd2_reg = 1.0 / (2.0 * PI * r_cub) + 1.0 / (4.0 * PI) * (1.0 - 9.0 * r / 8.0 + r_cub / 8.0);

        double fd2_term_p = fd2_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + fd2_erfpolyp * std::erfc((r + 2.0) * xi);
        double fd2_term_m = fd2_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + fd2_erfpolym * std::erfc((r - 2.0) * xi);
        double fd2_term_0 = fd2_exppoly0 * std::exp(-r_sq * xi_sq) + fd2_erfpoly0 * std::erfc(r * xi);
        double fd2_reg_part = (r < 2.0) ? fd2_reg : 0.0;

        fd2[idx] = fd2_term_p + fd2_term_m + fd2_term_0 + fd2_reg_part;

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

        fq1[idx] = fq1_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + fq1_erfpolyp * std::erfc((r + 2.0) * xi) +
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

        fq2[idx] = fq2_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + fq2_erfpolyp * std::erfc((r + 2.0) * xi) +
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

        fq3[idx] = fq3_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + fq3_erfpolyp * std::erfc((r + 2.0) * xi) +
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

        gq1[idx] = gq1_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + gq1_erfpolyp * std::erfc((r + 2.0) * xi) +
                   gq1_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + gq1_erfpolym * std::erfc((r - 2.0) * xi) +
                   gq1_exppoly0 * std::exp(-r_sq * xi_sq) + gq1_erfpoly0 * std::erfc(r * xi) +
                   ((r < 2.0) ? gq1_reg : 0.0);

        // --- grad_quad_2 calculation ---
        double gq2_exppolyp = 3.0 / (32768.0 * pi_pow_1_5 * xi_9 * r_5) * 
            (-48.0 * xi_8 * r_6 * r_cub + 96.0 * xi_8 * r_6 * r_sq - (576.0 * xi_6 - 608.0 * xi_8) * r_6 * r + (1056.0 * xi_6 - 1216.0 * xi_8) * r_6 - 
             (1536.0 * xi_4 - 2576.0 * xi_6 + 3968.0 * xi_8) * r_5 + (2160.0 * xi_4 - 4320.0 * xi_6 - 256.0 * xi_8) * r_4 - 
             (360.0 * xi_sq + 360.0 * xi_4 + 640.0 * xi_6 - 512.0 * xi_8) * r_cub - 
             (360.0 * xi_sq - 1680.0 * xi_4 - 768.0 * xi_6 + 1024.0 * xi_8) * r_sq + 
             (135.0 + 180.0 * xi_sq + 480.0 * xi_4 - 768.0 * xi_6 + 2048.0 * xi_8) * r + 
             (270.0 - 1080.0 * xi_sq - 192.0 * xi_4 + 512.0 * xi_6 - 4096.0 * xi_8));

        double gq2_exppolym = 3.0 / (32768.0 * pi_pow_1_5 * xi_9 * r_5) * 
            (-48.0 * xi_8 * r_6 * r_cub - 96.0 * xi_8 * r_6 * r_sq - (576.0 * xi_6 - 608.0 * xi_8) * r_6 * r - (1056.0 * xi_6 - 1216.0 * xi_8) * r_6 - 
             (1536.0 * xi_4 - 2576.0 * xi_6 + 3968.0 * xi_8) * r_5 - (2160.0 * xi_4 - 4320.0 * xi_6 - 256.0 * xi_8) * r_4 - 
             (360.0 * xi_sq + 360.0 * xi_4 + 640.0 * xi_6 - 512.0 * xi_8) * r_cub + 
             (360.0 * xi_sq - 1680.0 * xi_4 - 768.0 * xi_6 + 1024.0 * xi_8) * r_sq + 
             (135.0 + 180.0 * xi_sq + 480.0 * xi_4 - 768.0 * xi_6 + 2048.0 * xi_8) * r - 
             (270.0 - 1080.0 * xi_sq - 192.0 * xi_4 + 512.0 * xi_6 - 4096.0 * xi_8));

        double gq2_exppoly0 = 1.0 / (16384.0 * pi_pow_1_5 * xi_9 * r_4) * 
            (144.0 * xi_8 * r_8 + (1728.0 * xi_6 - 2400.0 * xi_8) * r_6 + (4608.0 * xi_4 - 13200.0 * xi_6 + 19200.0 * xi_8) * r_4 + 
             (1080.0 * xi_sq - 5400.0 * xi_4 + 19200.0 * xi_6) * r_sq - 405.0 + 2700.0 * xi_sq - 14400.0 * xi_4);

        double gq2_erfpolyp = 1.0 / (65536.0 * PI * xi_10 * r_5) * 
            (288.0 * xi_10 * r_10 + (3600.0 * xi_8 - 4800.0 * xi_10) * r_8 + (10800.0 * xi_6 - 28800.0 * xi_8 + 38400.0 * xi_10) * r_6 + 
             49152.0 * xi_10 * r_5 + (5400.0 * xi_4 - 21600.0 * xi_6 + 57600.0 * xi_8) * r_4 - 
             (1350.0 * xi_sq - 7200.0 * xi_4 + 28800.0 * xi_6) * r_sq + 405.0 - 2700.0 * xi_sq + 14400.0 * xi_4 + 49152.0 * xi_10);

        double gq2_erfpolym = 1.0 / (65536.0 * PI * xi_10 * r_5) * 
            (288.0 * xi_10 * r_10 + (3600.0 * xi_8 - 4800.0 * xi_10) * r_8 + (10800.0 * xi_6 - 28800.0 * xi_8 + 38400.0 * xi_10) * r_6 - 
             49152.0 * xi_10 * r_5 + (5400.0 * xi_4 - 21600.0 * xi_6 + 57600.0 * xi_8) * r_4 - 
             (1350.0 * xi_sq - 7200.0 * xi_4 + 28800.0 * xi_6) * r_sq + 405.0 - 2700.0 * xi_sq + 14400.0 * xi_4 + 49152.0 * xi_10);

        double gq2_erfpoly0 = -1.0 / (32768.0 * PI * xi_10 * r_5) * 
            (288.0 * xi_10 * r_10 + (3600.0 * xi_8 - 4800.0 * xi_10) * r_8 + (10800.0 * xi_6 - 28800.0 * xi_8 + 38400.0 * xi_10) * r_6 + 
             (5400.0 * xi_4 - 21600.0 * xi_6 + 57600.0 * xi_8) * r_4 - (1350.0 * xi_sq - 7200.0 * xi_4 + 28800.0 * xi_6) * r_sq + 
             405.0 - 2700.0 * xi_sq + 14400.0 * xi_4);

        double gq2_reg = -3.0 / (2.0 * PI) * (1.0 / r_5 - 1.0 + 25.0 * r / 32.0 - 25.0 * r_cub / 256.0 + 3.0 * r_5 / 512.0);

        gq2[idx] = gq2_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + gq2_erfpolyp * std::erfc((r + 2.0) * xi) +
                   gq2_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + gq2_erfpolym * std::erfc((r - 2.0) * xi) +
                   gq2_exppoly0 * std::exp(-r_sq * xi_sq) + gq2_erfpoly0 * std::erfc(r * xi) +
                   ((r < 2.0) ? gq2_reg : 0.0);

        // --- grad_quad_3 calculation ---
        double gq3_exppolyp = -15.0 / (65536.0 * pi_pow_1_5 * xi_9 * r_5) * 
            (48.0 * xi_8 * r_6 * r_cub - 96.0 * xi_8 * r_6 * r_sq + (336.0 * xi_6 - 288.0 * xi_8) * r_6 * r - 576.0 * (xi_6 - xi_8) * r_6 + 
             (216.0 * xi_4 + 144.0 * xi_6 + 128.0 * xi_8) * r_5 - (480.0 * xi_6 + 256.0 * xi_8) * r_4 - 
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

        gq3[idx] = gq3_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + gq3_erfpolyp * std::erfc((r + 2.0) * xi) +
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

        gq4[idx] = gq4_exppolyp * std::exp(-std::pow(r + 2.0, 2) * xi_sq) + gq4_erfpolyp * std::erfc((r + 2.0) * xi) +
                   gq4_exppolym * std::exp(-std::pow(r - 2.0, 2) * xi_sq) + gq4_erfpolym * std::erfc((r - 2.0) * xi) +
                   gq4_exppoly0 * std::exp(-r_sq * xi_sq) + gq4_erfpoly0 * std::erfc(r * xi) +
                   ((r < 2.0) ? gq4_reg : 0.0);
    }

    double selfo = (-1.0 + 6.0 * xi_sq + (1.0 - 2.0 * xi_sq) * std::exp(-4.0 * xi_sq)) / 
                   (16.0 * pi_pow_1_5 * xi_cub) + std::erfc(2.0 * xi) / (4.0 * PI);

    double self_grad = 0.5 * (1.0 / PI) * (3.0 * (3.0 - 10.0 * xi_sq + 20.0 * xi_4) / (8.0 * std::sqrt(PI) * xi_5) -
                       3.0 * (3.0 + 2.0 * xi_sq + 4.0 * xi_4) * std::exp(-4.0 * xi_sq) / (8.0 * std::sqrt(PI) * xi_5) + 3.0 * std::erfc(2.0 * xi));

    self_perp_val = selfo;
    self_G2_val = self_grad;

    table_size = 9001;
    std::vector<double> host_r_table(table_size);
    std::vector<double> host_field_dip_1(table_size);
    std::vector<double> host_field_dip_2(table_size);
    std::vector<double> host_field_quad_1(table_size);
    std::vector<double> host_field_quad_2(table_size);
    std::vector<double> host_field_quad_3(table_size);
    std::vector<double> host_grad_quad_1(table_size);
    std::vector<double> host_grad_quad_2(table_size);
    std::vector<double> host_grad_quad_3(table_size);
    std::vector<double> host_grad_quad_4(table_size);

    host_r_table[0] = 0.0;
    host_field_dip_1[0] = selfo;
    host_field_dip_2[0] = selfo;
    host_field_quad_1[0] = 0.0;
    host_field_quad_2[0] = 0.0;
    host_field_quad_3[0] = 0.0;
    host_grad_quad_1[0] = self_grad;
    host_grad_quad_2[0] = self_grad;
    host_grad_quad_3[0] = self_grad;
    host_grad_quad_4[0] = self_grad;

    for (size_t idx = 0; idx < num_r_steps; ++idx) {
        host_r_table[idx + 1] = r_vals[idx];
        host_field_dip_1[idx + 1] = fd1[idx];
        host_field_dip_2[idx + 1] = fd2[idx];
        host_field_quad_1[idx + 1] = fq1[idx];
        host_field_quad_2[idx + 1] = fq2[idx];
        host_field_quad_3[idx + 1] = fq3[idx];
        host_grad_quad_1[idx + 1] = gq1[idx];
        host_grad_quad_2[idx + 1] = gq2[idx];
        host_grad_quad_3[idx + 1] = gq3[idx];
        host_grad_quad_4[idx + 1] = gq4[idx];
    }

    // Allocate GPU memory
    size_t size_in_bytes = table_size * sizeof(double);
    if (d_r_table) CUDA_CHECK(cudaFree(d_r_table));
    if (d_field_dip_1) CUDA_CHECK(cudaFree(d_field_dip_1));
    if (d_field_dip_2) CUDA_CHECK(cudaFree(d_field_dip_2));
    if (d_field_quad_1) CUDA_CHECK(cudaFree(d_field_quad_1));
    if (d_field_quad_2) CUDA_CHECK(cudaFree(d_field_quad_2));
    if (d_field_quad_3) CUDA_CHECK(cudaFree(d_field_quad_3));
    if (d_grad_quad_1) CUDA_CHECK(cudaFree(d_grad_quad_1));
    if (d_grad_quad_2) CUDA_CHECK(cudaFree(d_grad_quad_2));
    if (d_grad_quad_3) CUDA_CHECK(cudaFree(d_grad_quad_3));
    if (d_grad_quad_4) CUDA_CHECK(cudaFree(d_grad_quad_4));

    CUDA_CHECK(cudaMalloc(&d_r_table, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_field_dip_1, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_field_dip_2, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_field_quad_1, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_field_quad_2, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_field_quad_3, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_quad_1, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_quad_2, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_quad_3, size_in_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_quad_4, size_in_bytes));

    // Copy host memory to device
    CUDA_CHECK(cudaMemcpy(d_r_table, host_r_table.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field_dip_1, host_field_dip_1.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field_dip_2, host_field_dip_2.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field_quad_1, host_field_quad_1.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field_quad_2, host_field_quad_2.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field_quad_3, host_field_quad_3.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_quad_1, host_grad_quad_1.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_quad_2, host_grad_quad_2.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_quad_3, host_grad_quad_3.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_quad_4, host_grad_quad_4.data(), size_in_bytes, cudaMemcpyHostToDevice));
}



void Monodisperse_Ewald_Electric_Field::computePrecalculations() {
    // 1. Calculate rc
    rc = std::sqrt(-std::log(errortol)) / xi;
    if (rc < 2.0) {
        rc = 2.0;
    }

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
                // Store [x, y, z]
                host_offset.push_back(x_off);
                host_offset.push_back(y_off);
                host_offset.push_back(z_off);

                // Store [x*spacing_x, y*spacing_y, z*spacing_z]
                host_offsetxyz.push_back(x_off * grid_spacing[0]);
                host_offsetxyz.push_back(y_off * grid_spacing[1]);
                host_offsetxyz.push_back(z_off * grid_spacing[2]);
            }
        }
    }

    num_offsets = host_offset.size() / 3;

    // 8. Scale Precalcs (reciprocal space coefficients)
    // Allocate device memory and copy
    if (!d_offset) {
        CUDA_CHECK(cudaMalloc(&d_offset, num_offsets * 3 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_offsetxyz, num_offsets * 3 * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(d_offset, host_offset.data(), num_offsets * 3 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsetxyz, host_offsetxyz.data(), num_offsets * 3 * sizeof(double), cudaMemcpyHostToDevice));

    computeScalePrecalcs();
}



void Monodisperse_Ewald_Electric_Field::getPrecalculationsHost(std::vector<int>& host_offset,
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
    if (use_recip_fp32) {
        std::vector<float> temp(grid_voxels);
        CUDA_CHECK(cudaMemcpy(temp.data(), d_scale_coef, grid_voxels * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < grid_voxels; ++i) host_scale_coef[i] = static_cast<double>(temp[i]);
    } else {
        CUDA_CHECK(cudaMemcpy(host_scale_coef.data(), d_scale_coef, grid_voxels * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(host_khat.data(), d_khat, grid_voxels * 3 * sizeof(double), cudaMemcpyDeviceToHost));
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

template <typename RecipReal>
__global__ void spread_precalcs_kernel(
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    const double* __restrict__ z_part,
    const int* __restrict__ offset,
    const double* __restrict__ offsetxyz,
    RecipReal* __restrict__ spread_coef,
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

    spread_idxs[idx] = (((gix + ox + 256) & 0x3FF) << 20) | (((giy + oy + 256) & 0x3FF) << 10) | ((giz + oz + 256) & 0x3FF);

    double gdx = dx + oxyz_x;
    double gdy = dy + oxyz_y;
    double gdz = dz + oxyz_z;

    double div_eta = (gdx * gdx / spectral_split_x) +
                     (gdy * gdy / spectral_split_y) +
                     (gdz * gdz / spectral_split_z);

    spread_coef[idx] = static_cast<RecipReal>(const_factor * exp(-2.0 * xi * xi * div_eta));
}

template <typename RecipReal>
__global__ void contract_precalcs_kernel(
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    const double* __restrict__ z_field,
    const int* __restrict__ offset,
    const double* __restrict__ offsetxyz,
    RecipReal* __restrict__ contract_coef,
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

    int geix = ((gix + ox) % num_grid_x + num_grid_x) % num_grid_x;
    int geiy = ((giy + oy) % num_grid_y + num_grid_y) % num_grid_y;
    int geiz = ((giz + oz) % num_grid_z + num_grid_z) % num_grid_z;

    int transposed_idx = o * num_field_points + i;
    particle_index[idx] = i;
    contract_idxs[transposed_idx] = geix * num_grid_y * num_grid_z + geiy * num_grid_z + geiz;

    double gdx = dx + oxyz_x;
    double gdy = dy + oxyz_y;
    double gdz = dz + oxyz_z;

    double div_eta = (gdx * gdx / spectral_split_x) +
                     (gdy * gdy / spectral_split_y) +
                     (gdz * gdz / spectral_split_z);

    contract_coef[transposed_idx] = static_cast<RecipReal>(const_factor * exp(-2.0 * xi * xi * div_eta));
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
    if (count > max_neighbors) count = max_neighbors;
    int start_idx = particle_offsets[i];

    double xi = x_part[i];
    double yi = y_part[i];
    double zi = z_part[i];

    for (int k = 0; k < count; ++k) {
        int j = neighbor_list[i * max_neighbors + k];

        double rx = xi - x_field[j];
        double ry = yi - y_field[j];
        double rz = zi - z_field[j];

        if (box_x > 0.0) rx -= box_x * round(rx / box_x);
        if (box_y > 0.0) ry -= box_y * round(ry / box_y);
        if (box_z > 0.0) rz -= box_z * round(rz / box_z);

        double d = sqrt(rx * rx + ry * ry + rz * rz);

        if (d < rc) {
            double d_eff = d < 1.0 ? 1.0 : d;
            double p_val = interpolate_table_gpu(d_eff, r_table, field_dip_1, table_size);
            double a_val = interpolate_table_gpu(d_eff, r_table, field_dip_2, table_size);

            perp[start_idx + k] = p_val;
            para[start_idx + k] = a_val;
        }
    }
}


__global__ void real_space_precalcs_kernel_joint(
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
    const double* __restrict__ field_quad_1,
    const double* __restrict__ field_quad_2,
    const double* __restrict__ field_quad_3,
    const double* __restrict__ grad_quad_1,
    const double* __restrict__ grad_quad_2,
    const double* __restrict__ grad_quad_3,
    const double* __restrict__ grad_quad_4,
    size_t table_size,
    double* __restrict__ perp,
    double* __restrict__ para,
    double* __restrict__ perp_Q,
    double* __restrict__ para_Q,
    double* __restrict__ Q3,
    double* __restrict__ G1,
    double* __restrict__ G2,
    double* __restrict__ G3,
    double* __restrict__ G4,
    size_t num_particles,
    int max_neighbors,
    double box_x,
    double box_y,
    double box_z,
    double rc,
    bool solve_quadrupoles)
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

        double rx = xi - x_field[j];
        double ry = yi - y_field[j];
        double rz = zi - z_field[j];

        if (box_x > 0.0) rx -= box_x * round(rx / box_x);
        if (box_y > 0.0) ry -= box_y * round(ry / box_y);
        if (box_z > 0.0) rz -= box_z * round(rz / box_z);

        double d = sqrt(rx * rx + ry * ry + rz * rz);

        if (d < rc) {
            double d_eff = d < 1.0 ? 1.0 : d;
            double p_val = interpolate_table_gpu(d_eff, r_table, field_dip_1, table_size);
            double a_val = interpolate_table_gpu(d_eff, r_table, field_dip_2, table_size);

            perp[start_idx + k] = p_val;
            para[start_idx + k] = a_val;

            if (solve_quadrupoles) {
                perp_Q[start_idx + k] = interpolate_table_gpu(d_eff, r_table, field_quad_1, table_size);
                para_Q[start_idx + k] = interpolate_table_gpu(d_eff, r_table, field_quad_2, table_size);
                Q3[start_idx + k]     = interpolate_table_gpu(d_eff, r_table, field_quad_3, table_size);
                G1[start_idx + k]     = interpolate_table_gpu(d_eff, r_table, grad_quad_1, table_size);
                G2[start_idx + k]     = interpolate_table_gpu(d_eff, r_table, grad_quad_2, table_size);
                G3[start_idx + k]     = interpolate_table_gpu(d_eff, r_table, grad_quad_3, table_size);
                G4[start_idx + k]     = interpolate_table_gpu(d_eff, r_table, grad_quad_4, table_size);
            }
        }
    }
}

void Monodisperse_Ewald_Electric_Field::spreadPrecalcs() {
    if (num_particles == 0) return;

    num_spread = num_particles * num_offsets;
    size_t element_size = use_recip_fp32 ? sizeof(float) : sizeof(double);
    size_t size_coef_bytes = num_spread * element_size;
    size_t size_idxs_bytes = num_spread * sizeof(int);

    if (d_spread_coef) CUDA_CHECK(cudaFree(d_spread_coef));
    if (d_spread_idxs) CUDA_CHECK(cudaFree(d_spread_idxs));

    CUDA_CHECK(cudaMalloc(&d_spread_coef, size_coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_spread_idxs, size_idxs_bytes));

    const double PI = 3.14159265358979323846;
    double prod_split = spectral_split[0] * spectral_split[1] * spectral_split[2];
    double const_factor = std::pow(2.0 * xi * xi / PI, 1.5) * std::sqrt(1.0 / prod_split);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_spread + threadsPerBlock - 1) / threadsPerBlock;

    if (use_recip_fp32) {
        spread_precalcs_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
            d_x_part, d_y_part, d_z_part,
            d_offset, d_offsetxyz,
            static_cast<float*>(d_spread_coef), d_spread_idxs,
            num_particles, num_offsets,
            grid_spacing[0], grid_spacing[1], grid_spacing[2],
            num_grid[0], num_grid[1], num_grid[2],
            spectral_split[0], spectral_split[1], spectral_split[2],
            const_factor, xi
        );
    } else {
        spread_precalcs_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(
            d_x_part, d_y_part, d_z_part,
            d_offset, d_offsetxyz,
            static_cast<double*>(d_spread_coef), d_spread_idxs,
            num_particles, num_offsets,
            grid_spacing[0], grid_spacing[1], grid_spacing[2],
            num_grid[0], num_grid[1], num_grid[2],
            spectral_split[0], spectral_split[1], spectral_split[2],
            const_factor, xi
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Monodisperse_Ewald_Electric_Field::contractPrecalcs() {
    if (num_field_points == 0) return;

    num_contract = num_field_points * num_offsets;
    size_t element_size = use_recip_fp32 ? sizeof(float) : sizeof(double);
    size_t size_coef_bytes = num_contract * element_size;
    size_t size_idxs_bytes = num_contract * sizeof(int);
    size_t size_part_idx_bytes = num_contract * sizeof(int);
    size_t size_epoint_bytes = (num_field_points * 3 + num_quads * 5) * 2 * sizeof(double);

    if (d_E_point) CUDA_CHECK(cudaFree(d_E_point));
    if (d_particle_index) CUDA_CHECK(cudaFree(d_particle_index));
    if (d_contract_coef) CUDA_CHECK(cudaFree(d_contract_coef));
    if (d_contract_idxs) CUDA_CHECK(cudaFree(d_contract_idxs));
    if (d_G_point) { CUDA_CHECK(cudaFree(d_G_point)); d_G_point = nullptr; }

    CUDA_CHECK(cudaMalloc(&d_E_point, size_epoint_bytes));
    CUDA_CHECK(cudaMemset(d_E_point, 0, size_epoint_bytes));

    CUDA_CHECK(cudaMalloc(&d_particle_index, size_part_idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_contract_coef, size_coef_bytes));
    CUDA_CHECK(cudaMalloc(&d_contract_idxs, size_idxs_bytes));

    if (solve_quadrupoles) {
        CUDA_CHECK(cudaMalloc(&d_G_point, num_field_points * 5 * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_G_point, 0, num_field_points * 5 * 2 * sizeof(double)));
    }

    const double PI = 3.14159265358979323846;
    double prod_split = spectral_split[0] * spectral_split[1] * spectral_split[2];
    double prod_spacing = grid_spacing[0] * grid_spacing[1] * grid_spacing[2];
    double const_factor = std::pow(2.0 * xi * xi / PI, 1.5) * std::sqrt(1.0 / prod_split) * prod_spacing;

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_contract + threadsPerBlock - 1) / threadsPerBlock;

    if (use_recip_fp32) {
        contract_precalcs_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
            d_x_field, d_y_field, d_z_field,
            d_offset, d_offsetxyz,
            static_cast<float*>(d_contract_coef), d_contract_idxs, d_particle_index,
            num_field_points, num_offsets,
            grid_spacing[0], grid_spacing[1], grid_spacing[2],
            num_grid[0], num_grid[1], num_grid[2],
            spectral_split[0], spectral_split[1], spectral_split[2],
            const_factor, xi
        );
    } else {
        contract_precalcs_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(
            d_x_field, d_y_field, d_z_field,
            d_offset, d_offsetxyz,
            static_cast<double*>(d_contract_coef), d_contract_idxs, d_particle_index,
            num_field_points, num_offsets,
            grid_spacing[0], grid_spacing[1], grid_spacing[2],
            num_grid[0], num_grid[1], num_grid[2],
            spectral_split[0], spectral_split[1], spectral_split[2],
            const_factor, xi
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Monodisperse_Ewald_Electric_Field::realSpacePrecalcs() {
    computeNeighborList(128);

    size_t num_pairs = neighbor_list->get_num_pairs();

    if (d_self_perp) CUDA_CHECK(cudaFree(d_self_perp));
    if (d_perp) CUDA_CHECK(cudaFree(d_perp));
    if (d_para) CUDA_CHECK(cudaFree(d_para));

    if (d_perp_Q) { cudaFree(d_perp_Q); d_perp_Q = nullptr; }
    if (d_para_Q) { cudaFree(d_para_Q); d_para_Q = nullptr; }
    if (d_Q3) { cudaFree(d_Q3); d_Q3 = nullptr; }
    if (d_G1) { cudaFree(d_G1); d_G1 = nullptr; }
    if (d_G2) { cudaFree(d_G2); d_G2 = nullptr; }
    if (d_G3) { cudaFree(d_G3); d_G3 = nullptr; }
    if (d_G4) { cudaFree(d_G4); d_G4 = nullptr; }

    CUDA_CHECK(cudaMalloc(&d_self_perp, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_self_perp, &self_perp_val, sizeof(double), cudaMemcpyHostToDevice));

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

        real_space_precalcs_kernel_joint<<<blocksPerGrid, threadsPerBlock>>>(
            d_x_part, d_y_part, d_z_part,
            d_x_field, d_y_field, d_z_field,
            neighbor_list->get_list(), neighbor_list->get_counts(), neighbor_list->get_offsets(),
            d_r_table, d_field_dip_1, d_field_dip_2,
            d_field_quad_1, d_field_quad_2, d_field_quad_3,
            d_grad_quad_1, d_grad_quad_2, d_grad_quad_3, d_grad_quad_4,
            table_size,
            d_perp, d_para,
            d_perp_Q, d_para_Q, d_Q3, d_G1, d_G2, d_G3, d_G4,
            num_particles,
            neighbor_list->get_max_neighbors(),
            box_x, box_y, box_z,
            rc,
            solve_quadrupoles
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());


    } else {
        d_perp = nullptr;
        d_para = nullptr;
    }
}





void Monodisperse_Ewald_Electric_Field::getRealSpacePrecalcsHost(double& host_self_perp,
                                              std::vector<double>& host_perp,
                                              std::vector<double>& host_para) const {
    if (d_self_perp == nullptr) {
        throw std::runtime_error("Real space precalcs have not been allocated on GPU yet.");
    }

    CUDA_CHECK(cudaMemcpy(&host_self_perp, d_self_perp, sizeof(double), cudaMemcpyDeviceToHost));

    size_t num_pairs = neighbor_list ? neighbor_list->get_num_pairs() : 0;
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

template <typename RecipReal>
__global__ void spread_kernel(
    const double* __restrict__ d_dipoles,
    const RecipReal* __restrict__ spread_coef,
    const int* __restrict__ spread_idxs,
    RecipReal* __restrict__ fE_grid,
    size_t num_spread,
    size_t num_offsets,
    int num_grid_x,
    int num_grid_y,
    int num_grid_z,
    const double* __restrict__ x_part,
    const double* __restrict__ y_part,
    double k_x,
    double k_y)
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
    RecipReal coef = 0.0;
    double dx_r = 0.0, dx_i = 0.0;
    double dy_r = 0.0, dy_i = 0.0;
    double dz_r = 0.0, dz_i = 0.0;

    if (active) {
        int i = idx / num_offsets;

        const double2* d_dipoles_d2 = reinterpret_cast<const double2*>(d_dipoles);
        double2 dip_x = d_dipoles_d2[i * 3 + 0];
        double2 dip_y = d_dipoles_d2[i * 3 + 1];
        double2 dip_z = d_dipoles_d2[i * 3 + 2];

        dx_r = dip_x.x;
        dx_i = dip_x.y;
        dy_r = dip_y.x;
        dy_i = dip_y.y;
        dz_r = dip_z.x;
        dz_i = dip_z.y;

        coef = spread_coef[idx];

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

    extern __shared__ char s_grid_raw[];
    RecipReal* s_grid = reinterpret_cast<RecipReal*>(s_grid_raw);

    bool use_shared = (local_grid_size > 0 && local_grid_size <= 512);

    if (use_shared) {
        // Initialize shared memory
        for (int offset = threadIdx.x; offset < local_grid_size * 6; offset += blockDim.x) {
            s_grid[offset] = 0.0;
        }
        __syncthreads();

        if (active) {
            int local_x = gx - block_min_gx;
            int local_y = gy - block_min_gy;
            int local_z = gz - block_min_gz;
            int local_idx = local_x * dim_y * dim_z + local_y * dim_z + local_z;

            atomicAdd(&s_grid[(local_idx * 3 + 0) * 2 + 0], static_cast<RecipReal>(coef * dx_r));
            atomicAdd(&s_grid[(local_idx * 3 + 0) * 2 + 1], static_cast<RecipReal>(coef * dx_i));
            atomicAdd(&s_grid[(local_idx * 3 + 1) * 2 + 0], static_cast<RecipReal>(coef * dy_r));
            atomicAdd(&s_grid[(local_idx * 3 + 1) * 2 + 1], static_cast<RecipReal>(coef * dy_i));
            atomicAdd(&s_grid[(local_idx * 3 + 2) * 2 + 0], static_cast<RecipReal>(coef * dz_r));
            atomicAdd(&s_grid[(local_idx * 3 + 2) * 2 + 1], static_cast<RecipReal>(coef * dz_i));
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

            size_t global_idx = (static_cast<size_t>(global_gx) * num_grid_y * num_grid_z +
                                  static_cast<size_t>(global_gy) * num_grid_z +
                                  static_cast<size_t>(global_gz)) * 3;

            RecipReal val_x_r = s_grid[(offset * 3 + 0) * 2 + 0];
            RecipReal val_x_i = s_grid[(offset * 3 + 0) * 2 + 1];
            RecipReal val_y_r = s_grid[(offset * 3 + 1) * 2 + 0];
            RecipReal val_y_i = s_grid[(offset * 3 + 1) * 2 + 1];
            RecipReal val_z_r = s_grid[(offset * 3 + 2) * 2 + 0];
            RecipReal val_z_i = s_grid[(offset * 3 + 2) * 2 + 1];

            if (val_x_r != 0.0) atomicAdd(&fE_grid[(global_idx + 0) * 2 + 0], val_x_r);
            if (val_x_i != 0.0) atomicAdd(&fE_grid[(global_idx + 0) * 2 + 1], val_x_i);
            if (val_y_r != 0.0) atomicAdd(&fE_grid[(global_idx + 1) * 2 + 0], val_y_r);
            if (val_y_i != 0.0) atomicAdd(&fE_grid[(global_idx + 1) * 2 + 1], val_y_i);
            if (val_z_r != 0.0) atomicAdd(&fE_grid[(global_idx + 2) * 2 + 0], val_z_r);
            if (val_z_i != 0.0) atomicAdd(&fE_grid[(global_idx + 2) * 2 + 1], val_z_i);
        }
    } else {
        // Fallback to direct global memory writes
        if (active) {
            int global_gx = ((gx) % num_grid_x + num_grid_x) % num_grid_x;
            int global_gy = ((gy) % num_grid_y + num_grid_y) % num_grid_y;
            int global_gz = ((gz) % num_grid_z + num_grid_z) % num_grid_z;

            size_t global_idx = (static_cast<size_t>(global_gx) * num_grid_y * num_grid_z +
                                  static_cast<size_t>(global_gy) * num_grid_z +
                                  static_cast<size_t>(global_gz)) * 3;

            atomicAdd(&fE_grid[(global_idx + 0) * 2 + 0], static_cast<RecipReal>(coef * dx_r));
            atomicAdd(&fE_grid[(global_idx + 0) * 2 + 1], static_cast<RecipReal>(coef * dx_i));
            atomicAdd(&fE_grid[(global_idx + 1) * 2 + 0], static_cast<RecipReal>(coef * dy_r));
            atomicAdd(&fE_grid[(global_idx + 1) * 2 + 1], static_cast<RecipReal>(coef * dy_i));
            atomicAdd(&fE_grid[(global_idx + 2) * 2 + 0], static_cast<RecipReal>(coef * dz_r));
            atomicAdd(&fE_grid[(global_idx + 2) * 2 + 1], static_cast<RecipReal>(coef * dz_i));
        }
    }
}

template <typename RecipReal>
__global__ void compute_scale_coefficients_kernel(
    RecipReal* d_scale_coef,
    double* d_khat,
    RecipReal* d_scale_coef_Q_imag,
    RecipReal* d_scale_coef_GP_imag,
    RecipReal* d_scale_coef_GQ_real,
    RecipReal* d_Qfactor,
    RecipReal* d_Qfactor_dot,
    int num_grid_x, int num_grid_y, int num_grid_z,
    double box_x, double box_y, double box_z,
    double k_x, double k_y,
    double xi,
    double spectral_split_x, double spectral_split_y, double spectral_split_z,
    bool solve_quadrupoles,
    size_t grid_voxels)
{
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_idx >= grid_voxels) return;

    int ix = linear_idx / (num_grid_y * num_grid_z);
    int iy = (linear_idx / num_grid_z) % num_grid_y;
    int iz = linear_idx % num_grid_z;

    double freq_x = (ix <= (num_grid_x - 1) / 2) ? ix : (ix - num_grid_x);
    double freq_y = (iy <= (num_grid_y - 1) / 2) ? iy : (iy - num_grid_y);
    double freq_z = (iz <= (num_grid_z - 1) / 2) ? iz : (iz - num_grid_z);

    const double PI = 3.14159265358979323846;
    double kx_val = freq_x * 2.0 * PI / box_x - k_x;
    double ky_val = freq_y * 2.0 * PI / box_y - k_y;
    double kz_val = freq_z * 2.0 * PI / box_z;

    double ksqx = kx_val * kx_val;
    double ksqy = ky_val * ky_val;
    double ksqz = kz_val * kz_val;

    double ksqsm = ksqx + ksqy + ksqz;
    double kmag = sqrt(ksqsm);

    if (kmag < 1e-12) {
        d_scale_coef[linear_idx] = 0.0;
        d_khat[linear_idx * 3 + 0] = 0.0;
        d_khat[linear_idx * 3 + 1] = 0.0;
        d_khat[linear_idx * 3 + 2] = 0.0;
        if (solve_quadrupoles) {
            d_scale_coef_Q_imag[linear_idx] = 0.0;
            d_scale_coef_GP_imag[linear_idx] = 0.0;
            d_scale_coef_GQ_real[linear_idx] = 0.0;
            for (int c = 0; c < 5; ++c) {
                d_Qfactor[linear_idx * 5 + c] = 0.0;
                d_Qfactor_dot[linear_idx * 5 + c] = 0.0;
            }
        }
        return;
    }

    double kh0 = kx_val / kmag;
    double kh1 = ky_val / kmag;
    double kh2 = kz_val / kmag;

    d_khat[linear_idx * 3 + 0] = kh0;
    d_khat[linear_idx * 3 + 1] = kh1;
    d_khat[linear_idx * 3 + 2] = kh2;

    if (solve_quadrupoles) {
        d_Qfactor[linear_idx * 5 + 0] = static_cast<RecipReal>(kh0 * kh0 - 1.0 / 3.0);
        d_Qfactor[linear_idx * 5 + 1] = static_cast<RecipReal>(kh0 * kh1);
        d_Qfactor[linear_idx * 5 + 2] = static_cast<RecipReal>(kh0 * kh2);
        d_Qfactor[linear_idx * 5 + 3] = static_cast<RecipReal>(kh1 * kh1 - 1.0 / 3.0);
        d_Qfactor[linear_idx * 5 + 4] = static_cast<RecipReal>(kh1 * kh2);

        d_Qfactor_dot[linear_idx * 5 + 0] = static_cast<RecipReal>(kh0 * kh0 - kh2 * kh2);
        d_Qfactor_dot[linear_idx * 5 + 1] = static_cast<RecipReal>(2.0 * kh0 * kh1);
        d_Qfactor_dot[linear_idx * 5 + 2] = static_cast<RecipReal>(2.0 * kh0 * kh2);
        d_Qfactor_dot[linear_idx * 5 + 3] = static_cast<RecipReal>(kh1 * kh1 - kh2 * kh2);
        d_Qfactor_dot[linear_idx * 5 + 4] = static_cast<RecipReal>(2.0 * kh1 * kh2);
    }

    double Gx_val = freq_x * 2.0 * PI / box_x;
    double Gy_val = freq_y * 2.0 * PI / box_y;
    double Gz_val = freq_z * 2.0 * PI / box_z;
    double Gx_sq = Gx_val * Gx_val;
    double Gy_sq = Gy_val * Gy_val;
    double Gz_sq = Gz_val * Gz_val;

    double etaksq = ksqsm - (Gx_sq * spectral_split_x + 
                             Gy_sq * spectral_split_y + 
                             Gz_sq * spectral_split_z);

    double term = sin(kmag) / kmag - cos(kmag);
    double exp_part = exp(-etaksq / (4.0 * xi * xi));
    double scale_factor = 1.0 / static_cast<double>(grid_voxels);
    d_scale_coef[linear_idx] = static_cast<RecipReal>((9.0 * term * term * exp_part / (ksqsm * ksqsm)) * scale_factor);

    if (solve_quadrupoles) {
        double j1 = term / kmag;
        double j2 = (3.0 / ksqsm - 1.0) * sin(kmag) / kmag - 3.0 * cos(kmag) / ksqsm;
        double expk2 = exp_part / ksqsm;

        d_scale_coef_Q_imag[linear_idx] = static_cast<RecipReal>((-22.5 * j1 * j2 * expk2) * scale_factor);
        d_scale_coef_GP_imag[linear_idx] = static_cast<RecipReal>((45.0 * j1 * j2 * expk2) * scale_factor);
        d_scale_coef_GQ_real[linear_idx] = static_cast<RecipReal>((112.5 * j2 * j2 * expk2) * scale_factor);
    }
}

template <typename RecipReal>
__global__ void scale_kernel(
    RecipReal* __restrict__ fE_grid,
    const RecipReal* __restrict__ scale_coef,
    const double* __restrict__ khat,
    size_t num_voxels)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;

    double kx = khat[v * 3 + 0];
    double ky = khat[v * 3 + 1];
    double kz = khat[v * 3 + 2];

    RecipReal sc = scale_coef[v];

    RecipReal fr0_r = fE_grid[(v * 3 + 0) * 2 + 0];
    RecipReal fr1_r = fE_grid[(v * 3 + 1) * 2 + 0];
    RecipReal fr2_r = fE_grid[(v * 3 + 2) * 2 + 0];

    RecipReal fr0_i = fE_grid[(v * 3 + 0) * 2 + 1];
    RecipReal fr1_i = fE_grid[(v * 3 + 1) * 2 + 1];
    RecipReal fr2_i = fE_grid[(v * 3 + 2) * 2 + 1];

    double dot_r = static_cast<double>(fr0_r) * kx + static_cast<double>(fr1_r) * ky + static_cast<double>(fr2_r) * kz;
    double dot_i = static_cast<double>(fr0_i) * kx + static_cast<double>(fr1_i) * ky + static_cast<double>(fr2_i) * kz;

    double sum_r = static_cast<double>(sc) * dot_r;
    double sum_i = static_cast<double>(sc) * dot_i;

    fE_grid[(v * 3 + 0) * 2 + 0] = static_cast<RecipReal>(kx * sum_r);
    fE_grid[(v * 3 + 0) * 2 + 1] = static_cast<RecipReal>(kx * sum_i);

    fE_grid[(v * 3 + 1) * 2 + 0] = static_cast<RecipReal>(ky * sum_r);
    fE_grid[(v * 3 + 1) * 2 + 1] = static_cast<RecipReal>(ky * sum_i);

    fE_grid[(v * 3 + 2) * 2 + 0] = static_cast<RecipReal>(kz * sum_r);
    fE_grid[(v * 3 + 2) * 2 + 1] = static_cast<RecipReal>(kz * sum_i);
}

template <typename RecipReal>
__global__ void scale_kernel_joint(
    RecipReal* __restrict__ fE_grid,
    RecipReal* __restrict__ fG_grid,
    const RecipReal* __restrict__ scale_coef,
    const RecipReal* __restrict__ scale_coef_Q_imag,
    const RecipReal* __restrict__ scale_coef_GP_imag,
    const RecipReal* __restrict__ scale_coef_GQ_real,
    const double* __restrict__ khat,
    const RecipReal* __restrict__ Qfactor,
    const RecipReal* __restrict__ Qfactor_dot,
    size_t num_voxels)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;

    double kx = khat[v * 3 + 0];
    double ky = khat[v * 3 + 1];
    double kz = khat[v * 3 + 2];

    double sc = static_cast<double>(scale_coef[v]);
    double sc_Q = static_cast<double>(scale_coef_Q_imag[v]);
    double sc_GP = static_cast<double>(scale_coef_GP_imag[v]);
    double sc_GQ = static_cast<double>(scale_coef_GQ_real[v]);

    double fr0_r = static_cast<double>(fE_grid[(v * 3 + 0) * 2 + 0]);
    double fr0_i = static_cast<double>(fE_grid[(v * 3 + 0) * 2 + 1]);
    double fr1_r = static_cast<double>(fE_grid[(v * 3 + 1) * 2 + 0]);
    double fr1_i = static_cast<double>(fE_grid[(v * 3 + 1) * 2 + 1]);
    double fr2_r = static_cast<double>(fE_grid[(v * 3 + 2) * 2 + 0]);
    double fr2_i = static_cast<double>(fE_grid[(v * 3 + 2) * 2 + 1]);

    double E_dot_k_R = fr0_r * kx + fr1_r * ky + fr2_r * kz;
    double E_dot_k_I = fr0_i * kx + fr1_i * ky + fr2_i * kz;

    double G_dot_Qdot_R = 0.0;
    double G_dot_Qdot_I = 0.0;
    for (int c = 0; c < 5; ++c) {
        double Qdot_c = static_cast<double>(Qfactor_dot[v * 5 + c]);
        G_dot_Qdot_R += static_cast<double>(fG_grid[(v * 5 + c) * 2 + 0]) * Qdot_c;
        G_dot_Qdot_I += static_cast<double>(fG_grid[(v * 5 + c) * 2 + 1]) * Qdot_c;
    }

    double Edot_E_R = sc * E_dot_k_R - sc_Q * G_dot_Qdot_I;
    double Edot_E_I = sc * E_dot_k_I + sc_Q * G_dot_Qdot_R;

    double Gdot_G_R = -sc_GP * E_dot_k_I + sc_GQ * G_dot_Qdot_R;
    double Gdot_G_I = sc_GP * E_dot_k_R + sc_GQ * G_dot_Qdot_I;

    fE_grid[(v * 3 + 0) * 2 + 0] = static_cast<RecipReal>(kx * Edot_E_R);
    fE_grid[(v * 3 + 0) * 2 + 1] = static_cast<RecipReal>(kx * Edot_E_I);
    fE_grid[(v * 3 + 1) * 2 + 0] = static_cast<RecipReal>(ky * Edot_E_R);
    fE_grid[(v * 3 + 1) * 2 + 1] = static_cast<RecipReal>(ky * Edot_E_I);
    fE_grid[(v * 3 + 2) * 2 + 0] = static_cast<RecipReal>(kz * Edot_E_R);
    fE_grid[(v * 3 + 2) * 2 + 1] = static_cast<RecipReal>(kz * Edot_E_I);

    for (int c = 0; c < 5; ++c) {
        double Qf_c = static_cast<double>(Qfactor[v * 5 + c]);
        fG_grid[(v * 5 + c) * 2 + 0] = static_cast<RecipReal>(Qf_c * Gdot_G_R);
        fG_grid[(v * 5 + c) * 2 + 1] = static_cast<RecipReal>(Qf_c * Gdot_G_I);
    }
}

template <typename RecipReal>
__global__ void contract_kernel(
    const RecipReal* __restrict__ Es_grid,
    const int* __restrict__ contract_idxs,
    const RecipReal* __restrict__ contract_coef,
    double* __restrict__ E_point,
    size_t num_field_points,
    size_t num_offsets,
    int num_grid_y,
    int num_grid_z,
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    double k_x,
    double k_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_field_points) return;

    double E_x_r = 0.0, E_x_i = 0.0;
    double E_y_r = 0.0, E_y_i = 0.0;
    double E_z_r = 0.0, E_z_i = 0.0;

    for (size_t o = 0; o < num_offsets; ++o) {
        size_t idx = o * num_field_points + i;
        int v = contract_idxs[idx];

        double coef = static_cast<double>(contract_coef[idx]);

        E_x_r += coef * static_cast<double>(Es_grid[(v * 3 + 0) * 2 + 0]);
        E_x_i += coef * static_cast<double>(Es_grid[(v * 3 + 0) * 2 + 1]);

        E_y_r += coef * static_cast<double>(Es_grid[(v * 3 + 1) * 2 + 0]);
        E_y_i += coef * static_cast<double>(Es_grid[(v * 3 + 1) * 2 + 1]);

        E_z_r += coef * static_cast<double>(Es_grid[(v * 3 + 2) * 2 + 0]);
        E_z_i += coef * static_cast<double>(Es_grid[(v * 3 + 2) * 2 + 1]);
    }

    atomicAdd(&E_point[(i * 3 + 0) * 2 + 0], E_x_r);
    atomicAdd(&E_point[(i * 3 + 0) * 2 + 1], E_x_i);
    atomicAdd(&E_point[(i * 3 + 1) * 2 + 0], E_y_r);
    atomicAdd(&E_point[(i * 3 + 1) * 2 + 1], E_y_i);
    atomicAdd(&E_point[(i * 3 + 2) * 2 + 0], E_z_r);
    atomicAdd(&E_point[(i * 3 + 2) * 2 + 1], E_z_i);
}

template <typename RecipReal>
__global__ void contract_kernel_G(
    const RecipReal* __restrict__ Gs_grid,
    const int* __restrict__ contract_idxs,
    const RecipReal* __restrict__ contract_coef,
    double* __restrict__ G_point,
    size_t num_field_points,
    size_t num_offsets,
    int num_grid_y,
    int num_grid_z,
    const double* __restrict__ x_field,
    const double* __restrict__ y_field,
    double k_x,
    double k_y)
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

        double coef = static_cast<double>(contract_coef[idx]);

        G_0_r += coef * static_cast<double>(Gs_grid[(v * 5 + 0) * 2 + 0]);
        G_0_i += coef * static_cast<double>(Gs_grid[(v * 5 + 0) * 2 + 1]);

        G_1_r += coef * static_cast<double>(Gs_grid[(v * 5 + 1) * 2 + 0]);
        G_1_i += coef * static_cast<double>(Gs_grid[(v * 5 + 1) * 2 + 1]);

        G_2_r += coef * static_cast<double>(Gs_grid[(v * 5 + 2) * 2 + 0]);
        G_2_i += coef * static_cast<double>(Gs_grid[(v * 5 + 2) * 2 + 1]);

        G_3_r += coef * static_cast<double>(Gs_grid[(v * 5 + 3) * 2 + 0]);
        G_3_i += coef * static_cast<double>(Gs_grid[(v * 5 + 3) * 2 + 1]);

        G_4_r += coef * static_cast<double>(Gs_grid[(v * 5 + 4) * 2 + 0]);
        G_4_i += coef * static_cast<double>(Gs_grid[(v * 5 + 4) * 2 + 1]);
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

__global__ void copy_G_to_E_kernel(
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
        atomicAdd(&dst[c], src[c]);
    }
}

__global__ void real_space_self_kernel_joint(
    const double* __restrict__ d_dipoles,
    const double* __restrict__ d_self_coef_r,
    const double* __restrict__ d_self_coef_i,
    const int* __restrict__ quad_idxs,
    const int* __restrict__ quad_map,
    double self_perp,
    double self_G2,
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

    double factor_r = sc_r + self_perp;
    double factor_i = sc_i;

    const double2* d_dipoles_d2 = reinterpret_cast<const double2*>(d_dipoles);
    double2 dip_x = d_dipoles_d2[i * 3 + 0];
    double2 dip_y = d_dipoles_d2[i * 3 + 1];
    double2 dip_z = d_dipoles_d2[i * 3 + 2];

    double dx_r = dip_x.x; double dx_i = dip_x.y;
    double dy_r = dip_y.x; double dy_i = dip_y.y;
    double dz_r = dip_z.x; double dz_i = dip_z.y;

    E_point[(i * 3 + 0) * 2 + 0] += factor_r * dx_r - factor_i * dx_i;
    E_point[(i * 3 + 0) * 2 + 1] += factor_r * dx_i + factor_i * dx_r;

    E_point[(i * 3 + 1) * 2 + 0] += factor_r * dy_r - factor_i * dy_i;
    E_point[(i * 3 + 1) * 2 + 1] += factor_r * dy_i + factor_i * dy_r;

    E_point[(i * 3 + 2) * 2 + 0] += factor_r * dz_r - factor_i * dz_i;
    E_point[(i * 3 + 2) * 2 + 1] += factor_r * dz_i + factor_i * dz_r;

    if (solve_quadrupoles) {
        int q = quad_map[i];
        if (q >= 0 && q < num_quads) {
            double q_sc_r = 2.5 * sc_r + 0.5 * self_G2;
            double q_sc_i = 2.5 * sc_i;

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

            G_point[(q * 5 + 0) * 2 + 0] += q_sc_r * q0_r - q_sc_i * q0_i;
            G_point[(q * 5 + 0) * 2 + 1] += q_sc_r * q0_i + q_sc_i * q0_r;

            G_point[(q * 5 + 1) * 2 + 0] += q_sc_r * q1_r - q_sc_i * q1_i;
            G_point[(q * 5 + 1) * 2 + 1] += q_sc_r * q1_i + q_sc_i * q1_r;

            G_point[(q * 5 + 2) * 2 + 0] += q_sc_r * q2_r - q_sc_i * q2_i;
            G_point[(q * 5 + 2) * 2 + 1] += q_sc_r * q2_i + q_sc_i * q2_r;

            G_point[(q * 5 + 3) * 2 + 0] += q_sc_r * q3_r - q_sc_i * q3_i;
            G_point[(q * 5 + 3) * 2 + 1] += q_sc_r * q3_i + q_sc_i * q3_r;

            G_point[(q * 5 + 4) * 2 + 0] += q_sc_r * q4_r - q_sc_i * q4_i;
            G_point[(q * 5 + 4) * 2 + 1] += q_sc_r * q4_i + q_sc_i * q4_r;
        }
    }
}

__global__ void real_space_neighbor_kernel_joint(
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
    FieldCalcMode mode,
    double k_x,
    double k_y)
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
    const double I_arr[5] = {1.0, 0.0, 0.0, 1.0, 0.0};

    for (int k = 0; k < count; ++k) {
        int j = neighbor_list[i * max_neighbors + k];

        double rx = xi - x_field[j];
        double ry = yi - y_field[j];
        double rz = zi - z_field[j];

        double Rx = 0.0, Ry = 0.0, Rz = 0.0;
        if (box_x > 0.0) { Rx = box_x * round(rx / box_x); rx -= Rx; }
        if (box_y > 0.0) { Ry = box_y * round(ry / box_y); ry -= Ry; }
        if (box_z > 0.0) { Rz = box_z * round(rz / box_z); rz -= Rz; }

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

            double rr_std[5];
            rr_std[0] = delta_x * delta_x;
            rr_std[1] = delta_x * delta_y;
            rr_std[2] = delta_x * delta_z;
            rr_std[3] = delta_y * delta_y;
            rr_std[4] = delta_y * delta_z;

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

            if (k_x != 0.0 || k_y != 0.0) {
                double phase = - (k_x * rx + k_y * ry);
                double c = cos(phase);
                double s = sin(phase);
                double t_r, t_i;

                t_r = E_dip_x_r * c - E_dip_x_i * s;
                t_i = E_dip_x_r * s + E_dip_x_i * c;
                E_dip_x_r = t_r; E_dip_x_i = t_i;

                t_r = E_dip_y_r * c - E_dip_y_i * s;
                t_i = E_dip_y_r * s + E_dip_y_i * c;
                E_dip_y_r = t_r; E_dip_y_i = t_i;

                t_r = E_dip_z_r * c - E_dip_z_i * s;
                t_i = E_dip_z_r * s + E_dip_z_i * c;
                E_dip_z_r = t_r; E_dip_z_i = t_i;
            }

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

                if (k_x != 0.0 || k_y != 0.0) {
                    double phase = - (k_x * rx + k_y * ry);
                    double c = cos(phase);
                    double s = sin(phase);
                    double t_r, t_i;

                    t_r = E_quad_x_r * c - E_quad_x_i * s;
                    t_i = E_quad_x_r * s + E_quad_x_i * c;
                    E_quad_x_r = t_r; E_quad_x_i = t_i;

                    t_r = E_quad_y_r * c - E_quad_y_i * s;
                    t_i = E_quad_y_r * s + E_quad_y_i * c;
                    E_quad_y_r = t_r; E_quad_y_i = t_i;

                    t_r = E_quad_z_r * c - E_quad_z_i * s;
                    t_i = E_quad_z_r * s + E_quad_z_i * c;
                    E_quad_z_r = t_r; E_quad_z_i = t_i;
                }
            }

            atomicAdd(&E_point[(j * 3 + 0) * 2 + 0], sign * (E_dip_x_r - E_quad_x_r));
            atomicAdd(&E_point[(j * 3 + 0) * 2 + 1], sign * (E_dip_x_i - E_quad_x_i));

            atomicAdd(&E_point[(j * 3 + 1) * 2 + 0], sign * (E_dip_y_r - E_quad_y_r));
            atomicAdd(&E_point[(j * 3 + 1) * 2 + 1], sign * (E_dip_y_i - E_quad_y_i));

            atomicAdd(&E_point[(j * 3 + 2) * 2 + 0], sign * (E_dip_z_r - E_quad_z_r));
            atomicAdd(&E_point[(j * 3 + 2) * 2 + 1], sign * (E_dip_z_i - E_quad_z_i));

            // 2. Contributions to G
            if (solve_quadrupoles) {
                int q_field = (j < num_particles) ? quad_map[j] : -1;
                if (q_field >= 0 && q_field < num_quads) {
                    double Q1_val = perp_Q[start_idx + k];
                    double Q2_val = para_Q[start_idx + k];
                    double Q3_val = Q3[start_idx + k];

                    double Pr_rP_r[5];
                    Pr_rP_r[0] = 2.0 * px_r * delta_x;
                    Pr_rP_r[1] = px_r * delta_y + py_r * delta_x;
                    Pr_rP_r[2] = px_r * delta_z + pz_r * delta_x;
                    Pr_rP_r[3] = 2.0 * py_r * delta_y;
                    Pr_rP_r[4] = py_r * delta_z + pz_r * delta_y;

                    double Pr_rP_i[5];
                    Pr_rP_i[0] = 2.0 * px_i * delta_x;
                    Pr_rP_i[1] = px_i * delta_y + py_i * delta_x;
                    Pr_rP_i[2] = px_i * delta_z + pz_i * delta_x;
                    Pr_rP_i[3] = 2.0 * py_i * delta_y;
                    Pr_rP_i[4] = py_i * delta_z + pz_i * delta_y;

                    double G_dip_r[5];
                    double G_dip_i[5];
                    for (int c = 0; c < 5; ++c) {
                        G_dip_r[c] = (Q1_val * rr_std[c] * r_P_r + Q2_val * Pr_rP_r[c] + (Q2_val + Q3_val) * I_arr[c] * r_P_r);
                        G_dip_i[c] = (Q1_val * rr_std[c] * r_P_i + Q2_val * Pr_rP_i[c] + (Q2_val + Q3_val) * I_arr[c] * r_P_i);
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
                        Q_rr_rr_Q_r[0] = 2.0 * Q_r_vec_r[0] * delta_x;
                        Q_rr_rr_Q_r[1] = Q_r_vec_r[0] * delta_y + Q_r_vec_r[1] * delta_x;
                        Q_rr_rr_Q_r[2] = Q_r_vec_r[0] * delta_z + Q_r_vec_r[2] * delta_x;
                        Q_rr_rr_Q_r[3] = 2.0 * Q_r_vec_r[1] * delta_y;
                        Q_rr_rr_Q_r[4] = Q_r_vec_r[1] * delta_z + Q_r_vec_r[2] * delta_y;

                        double Q_rr_rr_Q_i[5];
                        Q_rr_rr_Q_i[0] = 2.0 * Q_r_vec_i[0] * delta_x;
                        Q_rr_rr_Q_i[1] = Q_r_vec_i[0] * delta_y + Q_r_vec_i[1] * delta_x;
                        Q_rr_rr_Q_i[2] = Q_r_vec_i[0] * delta_z + Q_r_vec_i[2] * delta_x;
                        Q_rr_rr_Q_i[3] = 2.0 * Q_r_vec_i[1] * delta_y;
                        Q_rr_rr_Q_i[4] = Q_r_vec_i[1] * delta_z + Q_r_vec_i[2] * delta_y;

                        double q_val_r[5] = {q0_r, q1_r, q2_r, q3_r, q4_r};
                        double q_val_i[5] = {q0_i, q1_i, q2_i, q3_i, q4_i};

                        for (int c = 0; c < 5; ++c) {
                            G_quad_r[c] = 0.5 * (G1_val * I_arr[c] * Q__rr_r + G2_val * q_val_r[c] + G3_val * (I_arr[c] * Q__rr_r + 2.0 * Q_rr_rr_Q_r[c]) + G4_val * rr_std[c] * Q__rr_r);
                            G_quad_i[c] = 0.5 * (G1_val * I_arr[c] * Q__rr_i + G2_val * q_val_i[c] + G3_val * (I_arr[c] * Q__rr_i + 2.0 * Q_rr_rr_Q_i[c]) + G4_val * rr_std[c] * Q__rr_i);
                        }
                    }

                    if (k_x != 0.0 || k_y != 0.0) {
                        double phase = - (k_x * rx + k_y * ry);
                        double c = cos(phase);
                        double s = sin(phase);
                        for (int c_idx = 0; c_idx < 5; ++c_idx) {
                            double t_r, t_i;

                            t_r = G_dip_r[c_idx] * c - G_dip_i[c_idx] * s;
                            t_i = G_dip_r[c_idx] * s + G_dip_i[c_idx] * c;
                            G_dip_r[c_idx] = t_r; G_dip_i[c_idx] = t_i;

                            t_r = G_quad_r[c_idx] * c - G_quad_i[c_idx] * s;
                            t_i = G_quad_r[c_idx] * s + G_quad_i[c_idx] * c;
                            G_quad_r[c_idx] = t_r; G_quad_i[c_idx] = t_i;
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

template <typename RecipReal>
__global__ void spread_quadrupoles_kernel(
    const double* __restrict__ d_dipoles,
    const int* __restrict__ quad_idxs,
    const RecipReal* __restrict__ spread_coef,
    const int* __restrict__ spread_idxs,
    RecipReal* __restrict__ fG_grid,
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
    RecipReal coef = 0.0;
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

    extern __shared__ char s_grid_raw[];
    RecipReal* s_grid = reinterpret_cast<RecipReal*>(s_grid_raw);

    bool use_shared = (local_grid_size > 0 && local_grid_size <= 300);

    if (use_shared) {
        // Initialize shared memory
        for (int offset = threadIdx.x; offset < local_grid_size * 10; offset += blockDim.x) {
            s_grid[offset] = 0.0;
        }
        __syncthreads();

        if (active) {
            int local_x = gx - block_min_gx;
            int local_y = gy - block_min_gy;
            int local_z = gz - block_min_gz;
            int local_idx = local_x * dim_y * dim_z + local_y * dim_z + local_z;

            atomicAdd(&s_grid[(local_idx * 5 + 0) * 2 + 0], static_cast<RecipReal>(coef * q0_r));
            atomicAdd(&s_grid[(local_idx * 5 + 0) * 2 + 1], static_cast<RecipReal>(coef * q0_i));
            atomicAdd(&s_grid[(local_idx * 5 + 1) * 2 + 0], static_cast<RecipReal>(coef * q1_r));
            atomicAdd(&s_grid[(local_idx * 5 + 1) * 2 + 1], static_cast<RecipReal>(coef * q1_i));
            atomicAdd(&s_grid[(local_idx * 5 + 2) * 2 + 0], static_cast<RecipReal>(coef * q2_r));
            atomicAdd(&s_grid[(local_idx * 5 + 2) * 2 + 1], static_cast<RecipReal>(coef * q2_i));
            atomicAdd(&s_grid[(local_idx * 5 + 3) * 2 + 0], static_cast<RecipReal>(coef * q3_r));
            atomicAdd(&s_grid[(local_idx * 5 + 3) * 2 + 1], static_cast<RecipReal>(coef * q3_i));
            atomicAdd(&s_grid[(local_idx * 5 + 4) * 2 + 0], static_cast<RecipReal>(coef * q4_r));
            atomicAdd(&s_grid[(local_idx * 5 + 4) * 2 + 1], static_cast<RecipReal>(coef * q4_i));
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

            size_t global_idx = (static_cast<size_t>(global_gx) * num_grid_y * num_grid_z +
                                  static_cast<size_t>(global_gy) * num_grid_z +
                                  static_cast<size_t>(global_gz)) * 5;

            RecipReal val_0_r = s_grid[(offset * 5 + 0) * 2 + 0];
            RecipReal val_0_i = s_grid[(offset * 5 + 0) * 2 + 1];
            RecipReal val_1_r = s_grid[(offset * 5 + 1) * 2 + 0];
            RecipReal val_1_i = s_grid[(offset * 5 + 1) * 2 + 1];
            RecipReal val_2_r = s_grid[(offset * 5 + 2) * 2 + 0];
            RecipReal val_2_i = s_grid[(offset * 5 + 2) * 2 + 1];
            RecipReal val_3_r = s_grid[(offset * 5 + 3) * 2 + 0];
            RecipReal val_3_i = s_grid[(offset * 5 + 3) * 2 + 1];
            RecipReal val_4_r = s_grid[(offset * 5 + 4) * 2 + 0];
            RecipReal val_4_i = s_grid[(offset * 5 + 4) * 2 + 1];

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
        // Fallback to direct global memory writes
        if (active) {
            int global_gx = ((gx) % num_grid_x + num_grid_x) % num_grid_x;
            int global_gy = ((gy) % num_grid_y + num_grid_y) % num_grid_y;
            int global_gz = ((gz) % num_grid_z + num_grid_z) % num_grid_z;

            size_t global_idx = (static_cast<size_t>(global_gx) * num_grid_y * num_grid_z +
                                  static_cast<size_t>(global_gy) * num_grid_z +
                                  static_cast<size_t>(global_gz)) * 5;

            atomicAdd(&fG_grid[(global_idx + 0) * 2 + 0], static_cast<RecipReal>(coef * q0_r));
            atomicAdd(&fG_grid[(global_idx + 0) * 2 + 1], static_cast<RecipReal>(coef * q0_i));
            atomicAdd(&fG_grid[(global_idx + 1) * 2 + 0], static_cast<RecipReal>(coef * q1_r));
            atomicAdd(&fG_grid[(global_idx + 1) * 2 + 1], static_cast<RecipReal>(coef * q1_i));
            atomicAdd(&fG_grid[(global_idx + 2) * 2 + 0], static_cast<RecipReal>(coef * q2_r));
            atomicAdd(&fG_grid[(global_idx + 2) * 2 + 1], static_cast<RecipReal>(coef * q2_i));
            atomicAdd(&fG_grid[(global_idx + 3) * 2 + 0], static_cast<RecipReal>(coef * q3_r));
            atomicAdd(&fG_grid[(global_idx + 3) * 2 + 1], static_cast<RecipReal>(coef * q3_i));
            atomicAdd(&fG_grid[(global_idx + 4) * 2 + 0], static_cast<RecipReal>(coef * q4_r));
            atomicAdd(&fG_grid[(global_idx + 4) * 2 + 1], static_cast<RecipReal>(coef * q4_i));
        }
    }
}

void Monodisperse_Ewald_Electric_Field::spread(void* d_fE_grid_in) {
    if (num_spread == 0 || d_spread_coef == nullptr || d_spread_idxs == nullptr || d_fE_grid_in == nullptr) {
        throw std::runtime_error("spread: Buffers/Precalcs are not allocated.");
    }

    cudaStream_t s_recip = use_async_streams ? stream_recip : 0;
    size_t grid_voxels = static_cast<size_t>(num_grid[0]) * num_grid[1] * num_grid[2];
    size_t element_size = use_recip_fp32 ? sizeof(float) : sizeof(double);
    CUDA_CHECK(cudaMemsetAsync(d_fE_grid_in, 0, grid_voxels * 3 * 2 * element_size, s_recip));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_spread + threadsPerBlock - 1) / threadsPerBlock;

    if (use_recip_fp32) {
        spread_kernel<float><<<blocksPerGrid, threadsPerBlock, 24576, s_recip>>>(
            d_dipoles,
            static_cast<const float*>(d_spread_coef), d_spread_idxs,
            static_cast<float*>(d_fE_grid_in),
            num_spread, num_offsets,
            num_grid[0], num_grid[1], num_grid[2],
            d_x_part, d_y_part,
            k_x, k_y
        );
    } else {
        spread_kernel<double><<<blocksPerGrid, threadsPerBlock, 24576, s_recip>>>(
            d_dipoles,
            static_cast<const double*>(d_spread_coef), d_spread_idxs,
            static_cast<double*>(d_fE_grid_in),
            num_spread, num_offsets,
            num_grid[0], num_grid[1], num_grid[2],
            d_x_part, d_y_part,
            k_x, k_y
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

void Monodisperse_Ewald_Electric_Field::scale(void* d_fE_grid_in) {
    size_t grid_voxels = static_cast<size_t>(num_grid[0]) * num_grid[1] * num_grid[2];
    if (grid_voxels == 0 || d_scale_coef == nullptr || d_khat == nullptr || d_fE_grid_in == nullptr) {
        throw std::runtime_error("scale: Buffers/Precalcs are not allocated.");
    }

    cudaStream_t s_recip = use_async_streams ? stream_recip : 0;
    int threadsPerBlock = 256;
    int blocksPerGrid = (grid_voxels + threadsPerBlock - 1) / threadsPerBlock;

    if (use_recip_fp32) {
        if (solve_quadrupoles) {
            scale_kernel_joint<float><<<blocksPerGrid, threadsPerBlock, 0, s_recip>>>(
                static_cast<float*>(d_fE_grid_in),
                static_cast<float*>(d_fG_grid),
                static_cast<const float*>(d_scale_coef),
                static_cast<const float*>(d_scale_coef_Q_imag),
                static_cast<const float*>(d_scale_coef_GP_imag),
                static_cast<const float*>(d_scale_coef_GQ_real),
                d_khat,
                static_cast<const float*>(d_Qfactor),
                static_cast<const float*>(d_Qfactor_dot),
                grid_voxels
            );
        } else {
            scale_kernel<float><<<blocksPerGrid, threadsPerBlock, 0, s_recip>>>(
                static_cast<float*>(d_fE_grid_in),
                static_cast<const float*>(d_scale_coef),
                d_khat,
                grid_voxels
            );
        }
    } else {
        if (solve_quadrupoles) {
            scale_kernel_joint<double><<<blocksPerGrid, threadsPerBlock, 0, s_recip>>>(
                static_cast<double*>(d_fE_grid_in),
                static_cast<double*>(d_fG_grid),
                static_cast<const double*>(d_scale_coef),
                static_cast<const double*>(d_scale_coef_Q_imag),
                static_cast<const double*>(d_scale_coef_GP_imag),
                static_cast<const double*>(d_scale_coef_GQ_real),
                d_khat,
                static_cast<const double*>(d_Qfactor),
                static_cast<const double*>(d_Qfactor_dot),
                grid_voxels
            );
        } else {
            scale_kernel<double><<<blocksPerGrid, threadsPerBlock, 0, s_recip>>>(
                static_cast<double*>(d_fE_grid_in),
                static_cast<const double*>(d_scale_coef),
                d_khat,
                grid_voxels
            );
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void Monodisperse_Ewald_Electric_Field::contract(double* d_E_point, const void* d_Es_grid_in) {
    if (num_contract == 0 || d_particle_index == nullptr || d_contract_coef == nullptr ||
        d_contract_idxs == nullptr || d_E_point == nullptr || d_Es_grid_in == nullptr) {
        throw std::runtime_error("contract: Buffers/Precalcs are not allocated.");
    }

    cudaStream_t s_recip = use_async_streams ? stream_recip : 0;
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_field_points + threadsPerBlock - 1) / threadsPerBlock;

    if (use_recip_fp32) {
        contract_kernel<float><<<blocksPerGrid, threadsPerBlock, 0, s_recip>>>(
            static_cast<const float*>(d_Es_grid_in),
            d_contract_idxs,
            static_cast<const float*>(d_contract_coef),
            d_E_point,
            num_field_points,
            num_offsets,
            num_grid[1],
            num_grid[2],
            d_x_field, d_y_field,
            k_x, k_y
        );
    } else {
        contract_kernel<double><<<blocksPerGrid, threadsPerBlock, 0, s_recip>>>(
            static_cast<const double*>(d_Es_grid_in),
            d_contract_idxs,
            static_cast<const double*>(d_contract_coef),
            d_E_point,
            num_field_points,
            num_offsets,
            num_grid[1],
            num_grid[2],
            d_x_field, d_y_field,
            k_x, k_y
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

void Monodisperse_Ewald_Electric_Field::realSpace(double* d_E_point) {
    if (num_particles == 0 || d_E_point == nullptr) return;

    cudaStream_t s_real = use_async_streams ? stream_real : 0;

    if (mode == FieldCalcMode::SOLVER_AX) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

        real_space_self_kernel_joint<<<blocksPerGrid, threadsPerBlock, 0, s_real>>>(
            d_dipoles, d_self_coef_r, d_self_coef_i,
            d_quad_idxs, d_quad_map,
            self_perp_val,
            self_G2_val,
            d_E_point,
            num_particles,
            num_quads,
            solve_quadrupoles,
            mode
        );
        CUDA_CHECK(cudaGetLastError());
    }

    size_t num_pairs = neighbor_list ? neighbor_list->get_num_pairs() : 0;
    if (num_pairs > 0) {
        if (d_perp == nullptr || d_para == nullptr || neighbor_list->get_offsets() == nullptr) {
            throw std::runtime_error("realSpace: perp/para precalcs are not allocated.");
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

        real_space_neighbor_kernel_joint<<<blocksPerGrid, threadsPerBlock, 0, s_real>>>(
            d_x_part, d_y_part, d_z_part,
            d_x_field, d_y_field, d_z_field,
            d_dipoles,
            neighbor_list->get_list(), neighbor_list->get_counts(), neighbor_list->get_offsets(),
            d_quad_idxs, d_quad_map,
            d_perp, d_para,
            d_perp_Q, d_para_Q, d_Q3, d_G1, d_G2, d_G3, d_G4,
            d_E_point,
            num_particles,
            num_quads,
            neighbor_list->get_max_neighbors(),
            box_x, box_y, box_z,
            rc,
            solve_quadrupoles,
            mode,
            k_x,
            k_y
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void negate_vector_kernel(double* vec, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = -vec[idx];
    }
}

void Monodisperse_Ewald_Electric_Field::electricField() {
    if (d_E_point == nullptr) {
        throw std::runtime_error("electricField: d_E_point has not been allocated (contractPrecalcs not run).");
    }

    cudaStream_t s_real = use_async_streams ? stream_real : 0;
    cudaStream_t s_recip = use_async_streams ? stream_recip : 0;

    size_t size_epoint_bytes = (num_field_points * 3 + num_quads * 5) * 2 * sizeof(double);
    CUDA_CHECK(cudaMemsetAsync(d_E_point, 0, size_epoint_bytes, s_real));

    // Bind cuFFT to reciprocal stream
    if (fft_plan) cufftSetStream((cufftHandle)fft_plan, s_recip);
    if (fft_plan_G) cufftSetStream((cufftHandle)fft_plan_G, s_recip);

    // --- Reciprocal-space pipeline (on s_recip) ---
    spread(d_fE_grid);
    
    if (use_recip_fp32) {
        cufftResult plan_res = cufftExecC2C((cufftHandle)fft_plan,
                                           (cufftComplex*)d_fE_grid,
                                           (cufftComplex*)d_fE_grid,
                                           CUFFT_FORWARD);
        if (plan_res != CUFFT_SUCCESS) {
            throw std::runtime_error("cuFFT forward execution failed with code: " + std::to_string(plan_res));
        }
    } else {
        cufftResult plan_res = cufftExecZ2Z((cufftHandle)fft_plan,
                                           (cufftDoubleComplex*)d_fE_grid,
                                           (cufftDoubleComplex*)d_fE_grid,
                                           CUFFT_FORWARD);
        if (plan_res != CUFFT_SUCCESS) {
            throw std::runtime_error("cuFFT forward execution failed with code: " + std::to_string(plan_res));
        }
    }
    
    if (solve_quadrupoles && num_quads > 0 && d_fG_grid != nullptr) {
        size_t grid_voxels_Q = static_cast<size_t>(num_grid[0]) * num_grid[1] * num_grid[2];
        size_t element_size = use_recip_fp32 ? sizeof(float) : sizeof(double);
        CUDA_CHECK(cudaMemsetAsync(d_fG_grid, 0, grid_voxels_Q * 5 * 2 * element_size, s_recip));

        int threadsPerBlock = 256;
        size_t total_spread_Q = num_quads * num_offsets;
        int blocksPerGrid = (total_spread_Q + threadsPerBlock - 1) / threadsPerBlock;

        if (use_recip_fp32) {
            spread_quadrupoles_kernel<float><<<blocksPerGrid, threadsPerBlock, 24576, s_recip>>>(
                d_dipoles,
                d_quad_idxs,
                static_cast<const float*>(d_spread_coef), d_spread_idxs,
                static_cast<float*>(d_fG_grid),
                num_quads, num_particles, num_offsets,
                num_grid[0], num_grid[1], num_grid[2]
            );
            CUDA_CHECK(cudaGetLastError());
            
            cufftResult plan_res_G = cufftExecC2C((cufftHandle)fft_plan_G,
                                                 (cufftComplex*)d_fG_grid,
                                                 (cufftComplex*)d_fG_grid,
                                                 CUFFT_FORWARD);
            if (plan_res_G != CUFFT_SUCCESS) {
                throw std::runtime_error("cuFFT forward execution G failed with code: " + std::to_string(plan_res_G));
            }
        } else {
            spread_quadrupoles_kernel<double><<<blocksPerGrid, threadsPerBlock, 24576, s_recip>>>(
                d_dipoles,
                d_quad_idxs,
                static_cast<const double*>(d_spread_coef), d_spread_idxs,
                static_cast<double*>(d_fG_grid),
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
    }
    
    scale(d_fE_grid);
    
    if (use_recip_fp32) {
        cufftResult plan_res = cufftExecC2C((cufftHandle)fft_plan,
                                           (cufftComplex*)d_fE_grid,
                                           (cufftComplex*)d_fE_grid,
                                           CUFFT_INVERSE);
        if (plan_res != CUFFT_SUCCESS) {
            throw std::runtime_error("cuFFT inverse execution failed with code: " + std::to_string(plan_res));
        }
    } else {
        cufftResult plan_res = cufftExecZ2Z((cufftHandle)fft_plan,
                                           (cufftDoubleComplex*)d_fE_grid,
                                           (cufftDoubleComplex*)d_fE_grid,
                                           CUFFT_INVERSE);
        if (plan_res != CUFFT_SUCCESS) {
            throw std::runtime_error("cuFFT inverse execution failed with code: " + std::to_string(plan_res));
        }
    }
    
    if (solve_quadrupoles && num_quads > 0 && d_fG_grid != nullptr) {
        if (use_recip_fp32) {
            cufftResult plan_res_G = cufftExecC2C((cufftHandle)fft_plan_G,
                                                 (cufftComplex*)d_fG_grid,
                                                 (cufftComplex*)d_fG_grid,
                                                 CUFFT_INVERSE);
            if (plan_res_G != CUFFT_SUCCESS) {
                throw std::runtime_error("cuFFT inverse execution G failed with code: " + std::to_string(plan_res_G));
            }
        } else {
            cufftResult plan_res_G = cufftExecZ2Z((cufftHandle)fft_plan_G,
                                                 (cufftDoubleComplex*)d_fG_grid,
                                                 (cufftDoubleComplex*)d_fG_grid,
                                                 CUFFT_INVERSE);
            if (plan_res_G != CUFFT_SUCCESS) {
                throw std::runtime_error("cuFFT inverse execution G failed with code: " + std::to_string(plan_res_G));
            }
        }
    }
    
    contract(d_E_point, d_fE_grid);
    
    if (solve_quadrupoles && num_quads > 0 && d_G_point != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(d_G_point, 0, num_field_points * 5 * 2 * sizeof(double), s_recip));
        
        int contract_threads = 256;
        int contract_blocks = (num_field_points + contract_threads - 1) / contract_threads;
        if (use_recip_fp32) {
            contract_kernel_G<float><<<contract_blocks, contract_threads, 0, s_recip>>>(
                static_cast<const float*>(d_fG_grid),
                d_contract_idxs,
                static_cast<const float*>(d_contract_coef),
                d_G_point,
                num_field_points,
                num_offsets,
                num_grid[1],
                num_grid[2],
                d_x_field, d_y_field,
                k_x, k_y
            );
        } else {
            contract_kernel_G<double><<<contract_blocks, contract_threads, 0, s_recip>>>(
                static_cast<const double*>(d_fG_grid),
                d_contract_idxs,
                static_cast<const double*>(d_contract_coef),
                d_G_point,
                num_field_points,
                num_offsets,
                num_grid[1],
                num_grid[2],
                d_x_field, d_y_field,
                k_x, k_y
            );
        }
        CUDA_CHECK(cudaGetLastError());

        int copy_threads = 256;
        int copy_blocks = (num_quads + copy_threads - 1) / copy_threads;
        copy_G_to_E_kernel<<<copy_blocks, copy_threads, 0, s_recip>>>(
            d_G_point,
            d_quad_idxs,
            d_E_point,
            num_quads,
            num_field_points
        );
        CUDA_CHECK(cudaGetLastError());
    }

    if (use_async_streams) {
        CUDA_CHECK(cudaEventRecord(event_recip_done, s_recip));
    }

    // --- Real-space pipeline (on s_real) ---
    realSpace(d_E_point);

    // --- Synchronize real stream with reciprocal event ---
    if (use_async_streams) {
        CUDA_CHECK(cudaStreamWaitEvent(s_real, event_recip_done, 0));
    }

    if (mode == FieldCalcMode::INTERACTION_FIELD) {
        size_t num_targets = (num_field_points > 0) ? num_field_points : num_particles;
        size_t num_doubles_E = (num_targets * 3 + num_quads * 5) * 2;
        if (num_doubles_E > 0) {
            int threads = 256;
            int blocks = (num_doubles_E + threads - 1) / threads;
            negate_vector_kernel<<<blocks, threads, 0, s_real>>>(d_E_point, num_doubles_E);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (use_async_streams) {
        CUDA_CHECK(cudaStreamSynchronize(s_real));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void Monodisperse_Ewald_Electric_Field::computeScalePrecalcs() {
    size_t grid_voxels = static_cast<size_t>(num_grid[0]) * num_grid[1] * num_grid[2];
    size_t element_size = use_recip_fp32 ? sizeof(float) : sizeof(double);

    if (d_scale_coef) CUDA_CHECK(cudaFree(d_scale_coef));
    CUDA_CHECK(cudaMalloc(&d_scale_coef, grid_voxels * element_size));

    if (!d_khat) {
        CUDA_CHECK(cudaMalloc(&d_khat, grid_voxels * 3 * sizeof(double)));
    }

    if (solve_quadrupoles) {
        if (d_scale_coef_Q_imag) CUDA_CHECK(cudaFree(d_scale_coef_Q_imag));
        if (d_scale_coef_GP_imag) CUDA_CHECK(cudaFree(d_scale_coef_GP_imag));
        if (d_scale_coef_GQ_real) CUDA_CHECK(cudaFree(d_scale_coef_GQ_real));
        if (d_Qfactor) CUDA_CHECK(cudaFree(d_Qfactor));
        if (d_Qfactor_dot) CUDA_CHECK(cudaFree(d_Qfactor_dot));

        CUDA_CHECK(cudaMalloc(&d_scale_coef_Q_imag, grid_voxels * element_size));
        CUDA_CHECK(cudaMalloc(&d_scale_coef_GP_imag, grid_voxels * element_size));
        CUDA_CHECK(cudaMalloc(&d_scale_coef_GQ_real, grid_voxels * element_size));
        CUDA_CHECK(cudaMalloc(&d_Qfactor, grid_voxels * 5 * element_size));
        CUDA_CHECK(cudaMalloc(&d_Qfactor_dot, grid_voxels * 5 * element_size));
    } else {
        d_scale_coef_Q_imag = nullptr;
        d_scale_coef_GP_imag = nullptr;
        d_scale_coef_GQ_real = nullptr;
        d_Qfactor = nullptr;
        d_Qfactor_dot = nullptr;
    }

    int threads = 256;
    int blocks = (grid_voxels + threads - 1) / threads;

    if (use_recip_fp32) {
        compute_scale_coefficients_kernel<float><<<blocks, threads>>>(
            static_cast<float*>(d_scale_coef),
            d_khat,
            static_cast<float*>(d_scale_coef_Q_imag),
            static_cast<float*>(d_scale_coef_GP_imag),
            static_cast<float*>(d_scale_coef_GQ_real),
            static_cast<float*>(d_Qfactor),
            static_cast<float*>(d_Qfactor_dot),
            num_grid[0], num_grid[1], num_grid[2],
            box_x, box_y, box_z,
            k_x, k_y,
            xi,
            spectral_split[0], spectral_split[1], spectral_split[2],
            solve_quadrupoles,
            grid_voxels
        );
    } else {
        compute_scale_coefficients_kernel<double><<<blocks, threads>>>(
            static_cast<double*>(d_scale_coef),
            d_khat,
            static_cast<double*>(d_scale_coef_Q_imag),
            static_cast<double*>(d_scale_coef_GP_imag),
            static_cast<double*>(d_scale_coef_GQ_real),
            static_cast<double*>(d_Qfactor),
            static_cast<double*>(d_Qfactor_dot),
            num_grid[0], num_grid[1], num_grid[2],
            box_x, box_y, box_z,
            k_x, k_y,
            xi,
            spectral_split[0], spectral_split[1], spectral_split[2],
            solve_quadrupoles,
            grid_voxels
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Monodisperse_Ewald_Electric_Field::calculate() {
    if (kt_updated) {
        computeScalePrecalcs();
        kt_updated = false;
    }
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



