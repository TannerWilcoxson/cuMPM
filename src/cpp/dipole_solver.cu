#include "dipole_solver.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA Error Checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
      if (abort) exit(code);
   }
}

// -----------------------------------------------------------------------------
// CUDA Kernels for Vector Operations
// -----------------------------------------------------------------------------

__global__ void vector_add_kernel(double* y, const double* x, double alpha_r, double alpha_i, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    double xr = x[idx * 2 + 0];
    double xi = x[idx * 2 + 1];

    y[idx * 2 + 0] += alpha_r * xr - alpha_i * xi;
    y[idx * 2 + 1] += alpha_r * xi + alpha_i * xr;
}

__global__ void vector_scale_kernel(double* y, double alpha_r, double alpha_i, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    double yr = y[idx * 2 + 0];
    double yi = y[idx * 2 + 1];

    y[idx * 2 + 0] = alpha_r * yr - alpha_i * yi;
    y[idx * 2 + 1] = alpha_r * yi + alpha_i * yr;
}

__global__ void vector_sub_kernel(double* y, const double* x, const double* z, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    y[idx * 2 + 0] = x[idx * 2 + 0] - z[idx * 2 + 0];
    y[idx * 2 + 1] = x[idx * 2 + 1] - z[idx * 2 + 1];
}

__global__ void dot_product_kernel(const double* a, const double* b, double* out_r, double* out_i, size_t size) {
    __shared__ double sdata_r[256];
    __shared__ double sdata_i[256];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    double sum_r = 0.0;
    double sum_i = 0.0;

    if (i < size) {
        double ar = a[i * 2 + 0];
        double ai = a[i * 2 + 1];
        double br = b[i * 2 + 0];
        double bi = b[i * 2 + 1];

        // conj(a) * b = (ar - i*ai)*(br + i*bi) = (ar*br + ai*bi) + i*(ar*bi - ai*br)
        sum_r = ar * br + ai * bi;
        sum_i = ar * bi - ai * br;
    }

    sdata_r[tid] = sum_r;
    sdata_i[tid] = sum_i;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_r[tid] += sdata_r[tid + s];
            sdata_i[tid] += sdata_i[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_r, sdata_r[0]);
        atomicAdd(out_i, sdata_i[0]);
    }
}

__global__ void norm_kernel(const double* a, double* out, size_t size) {
    __shared__ double sdata[256];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;
    if (i < size) {
        double ar = a[i * 2 + 0];
        double ai = a[i * 2 + 1];
        sum = ar * ar + ai * ai;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// -----------------------------------------------------------------------------
// Host helpers for GPU vector operations
// -----------------------------------------------------------------------------

static void gpu_vector_add(double* d_y, const double* d_x, Complex alpha, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(d_y, d_x, alpha.real(), alpha.imag(), size);
    CUDA_CHECK(cudaGetLastError());
}

static void gpu_vector_scale(double* d_y, Complex alpha, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    vector_scale_kernel<<<blocks, threads>>>(d_y, alpha.real(), alpha.imag(), size);
    CUDA_CHECK(cudaGetLastError());
}

static void gpu_vector_sub(double* d_y, const double* d_x, const double* d_z, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    vector_sub_kernel<<<blocks, threads>>>(d_y, d_x, d_z, size);
    CUDA_CHECK(cudaGetLastError());
}

static Complex gpu_dot_product(const double* d_a, const double* d_b, size_t size, double* d_reduce_buf) {
    CUDA_CHECK(cudaMemset(d_reduce_buf, 0, 2 * sizeof(double)));
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dot_product_kernel<<<blocks, threads>>>(d_a, d_b, &d_reduce_buf[0], &d_reduce_buf[1], size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double host_res[2];
    CUDA_CHECK(cudaMemcpy(host_res, d_reduce_buf, 2 * sizeof(double), cudaMemcpyDeviceToHost));
    return Complex(host_res[0], host_res[1]);
}

static double gpu_norm(const double* d_a, size_t size, double* d_reduce_buf) {
    CUDA_CHECK(cudaMemset(d_reduce_buf, 0, sizeof(double)));
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    norm_kernel<<<blocks, threads>>>(d_a, d_reduce_buf, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double host_res = 0.0;
    CUDA_CHECK(cudaMemcpy(&host_res, d_reduce_buf, sizeof(double), cudaMemcpyDeviceToHost));
    return std::sqrt(host_res);
}

// -----------------------------------------------------------------------------
// Dipole_Solver Class Implementation
// -----------------------------------------------------------------------------

// Constructor 1: Full 2D eps_p
Dipole_Solver::Dipole_Solver(const std::vector<double>& box,
                             const std::vector<std::vector<Complex>>& eps_p,
                             const std::vector<double>& radius,
                             double eps_m,
                             double xi,
                             double tol,
                             bool quiet,
                             const std::string& guess_type)
    : box(box), eps_p(eps_p), radius(radius), eps_m(eps_m), xi(xi), tol(tol), quiet(quiet), guess_type(guess_type) {
}

// Constructor 2: Wavelength-dependent (1D eps_p)
Dipole_Solver::Dipole_Solver(const std::vector<double>& box,
                             const std::vector<Complex>& eps_p_1d,
                             const std::vector<double>& radius,
                             double eps_m,
                             double xi,
                             double tol,
                             bool quiet,
                             const std::string& guess_type)
    : box(box), radius(radius), eps_m(eps_m), xi(xi), tol(tol), quiet(quiet), guess_type(guess_type) {
    
    // We will expand eps_p_1d in compute() once we know num_particles
    this->eps_p.resize(eps_p_1d.size(), std::vector<Complex>(1, 0.0));
    for (size_t w = 0; w < eps_p_1d.size(); ++w) {
        this->eps_p[w][0] = eps_p_1d[w];
    }
}

// Constructor 3: Scalar eps_p
Dipole_Solver::Dipole_Solver(const std::vector<double>& box,
                             Complex eps_p_scalar,
                             const std::vector<double>& radius,
                             double eps_m,
                             double xi,
                             double tol,
                             bool quiet,
                             const std::string& guess_type)
    : box(box), radius(radius), eps_m(eps_m), xi(xi), tol(tol), quiet(quiet), guess_type(guess_type) {
    
    this->eps_p.resize(1, std::vector<Complex>(1, eps_p_scalar));
}

Dipole_Solver::~Dipole_Solver() {
    free_gpu_buffers();
}

void Dipole_Solver::print(const std::string& msg) const {
}

void Dipole_Solver::set_dims(size_t num_p) {
    num_particles = num_p;

    // Set default radius if empty or scalar
    if (radius.empty()) {
        radius.resize(num_particles, 1.0);
    } else if (radius.size() == 1) {
        double val = radius[0];
        radius.resize(num_particles, val);
    } else if (radius.size() != num_particles) {
        throw std::runtime_error("The number of particles is inconsistent with the number of radii provided.");
    }

    // Check if radii are identical
    for (size_t i = 1; i < num_particles; ++i) {
        if (radius[i] != radius[0]) {
            throw std::runtime_error("Radii of different sizes not yet supported.");
        }
    }

    // Expand eps_p if it was 1D or scalar
    if (eps_p.empty()) {
        throw std::runtime_error("eps_p is empty.");
    }
    num_wavevectors = eps_p.size();

    for (size_t w = 0; w < num_wavevectors; ++w) {
        if (eps_p[w].size() == 1) {
            Complex val = eps_p[w][0];
            eps_p[w].resize(num_particles, val);
        } else if (eps_p[w].size() != num_particles) {
            throw std::runtime_error("The number of particles is inconsistent with eps_p provided.");
        }
    }
}

void Dipole_Solver::nondimensionalize() {
    length_scale = radius[0];
    eps_scale = eps_m;

    box[0] /= length_scale;
    box[1] /= length_scale;
    box[2] /= length_scale;

    for (size_t i = 0; i < radius.size(); ++i) {
        radius[i] /= length_scale;
    }

    eps_m /= eps_scale; // becomes 1.0

    for (size_t w = 0; w < num_wavevectors; ++w) {
        for (size_t p = 0; p < num_particles; ++p) {
            eps_p[w][p] /= eps_scale;
        }
    }
}

void Dipole_Solver::calc_vol_frac() {
    double sum_r3 = 0.0;
    for (double r : radius) {
        sum_r3 += std::pow(r, 3.0);
    }
    double box_vol = box[0] * box[1] * box[2];
    vol_frac = (4.0 / 3.0) * M_PI * sum_r3 / box_vol;
}

void Dipole_Solver::precalculations() {
    calc_vol_frac();
}

void Dipole_Solver::allocate_gpu_buffers() {
    free_gpu_buffers();

    size_t vec_size = num_particles * 3;
    size_t size_bytes = vec_size * 2 * sizeof(double);
    size_t restart = std::min(vec_size, static_cast<size_t>(10));

    CUDA_CHECK(cudaMalloc(&d_x, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_r, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_w, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, (restart + 1) * size_bytes));
    CUDA_CHECK(cudaMalloc(&d_reduce_buf, 2 * sizeof(double)));
}

void Dipole_Solver::free_gpu_buffers() {
    if (d_x) { cudaFree(d_x); d_x = nullptr; }
    if (d_b) { cudaFree(d_b); d_b = nullptr; }
    if (d_r) { cudaFree(d_r); d_r = nullptr; }
    if (d_w) { cudaFree(d_w); d_w = nullptr; }
    if (d_V) { cudaFree(d_V); d_V = nullptr; }
    if (d_reduce_buf) { cudaFree(d_reduce_buf); d_reduce_buf = nullptr; }
}

// -----------------------------------------------------------------------------
// Guess Predictor Calculations
// -----------------------------------------------------------------------------

std::vector<Complex> Dipole_Solver::calc_guess(const std::vector<Complex>& prev_dip, size_t wavevec_idx) const {
    if (guess_type == "mean_field" || guess_type == "mean-field") {
        return calc_mean_field_guess(wavevec_idx);
    } else if (guess_type == "previous") {
        return calc_previous_guess(prev_dip, wavevec_idx);
    } else if (guess_type == "derivative") {
        return calc_derivative_guess(prev_dip, wavevec_idx);
    } else {
        throw std::runtime_error("Guess type " + guess_type + " not supported.");
    }
}

std::vector<Complex> Dipole_Solver::calc_mean_field_guess(size_t wavevec_idx) const {
    // Return guess of shape: num_particles * 3 * 3 (diagonal matrices)
    std::vector<Complex> dip_guess(num_particles * 9, 0.0);
    for (size_t p = 0; p < num_particles; ++p) {
        Complex beta = (eps_p[wavevec_idx][p] - 1.0) / (eps_p[wavevec_idx][p] + 2.0);
        Complex val = 4.0 * M_PI * beta / (1.0 - beta * vol_frac);
        
        dip_guess[p * 9 + 0 * 3 + 0] = val; // (0,0)
        dip_guess[p * 9 + 1 * 3 + 1] = val; // (1,1)
        dip_guess[p * 9 + 2 * 3 + 2] = val; // (2,2)
    }
    return dip_guess;
}

std::vector<Complex> Dipole_Solver::calc_previous_guess(const std::vector<Complex>& prev_dip, size_t wavevec_idx) const {
    if (wavevec_idx < 1) {
        return calc_mean_field_guess(wavevec_idx);
    }
    // prev_dip contains dipoles for previous wavevectors
    // Return frame_dips[wavevec_idx - 1]
    std::vector<Complex> dip_guess(num_particles * 9);
    size_t offset = (wavevec_idx - 1) * num_particles * 9;
    std::copy(prev_dip.begin() + offset, prev_dip.begin() + offset + num_particles * 9, dip_guess.begin());
    return dip_guess;
}

std::vector<Complex> Dipole_Solver::calc_derivative_guess(const std::vector<Complex>& prev_dip, size_t wavevec_idx) const {
    if (wavevec_idx < 2) {
        return calc_mean_field_guess(wavevec_idx);
    }
    size_t i = wavevec_idx;
    size_t im2 = i - 2;
    size_t im1 = i - 1;

    std::vector<Complex> dip_guess(num_particles * 9);
    for (size_t p = 0; p < num_particles; ++p) {
        Complex run = eps_p[im1][p] - eps_p[im2][p];
        Complex new_run = eps_p[i][p] - eps_p[im1][p];
        if (run == 0.0) {
            run = 1.0;
            new_run = 0.0;
        }

        for (int idx = 0; idx < 9; ++idx) {
            Complex val_im1 = prev_dip[im1 * num_particles * 9 + p * 9 + idx];
            Complex val_im2 = prev_dip[im2 * num_particles * 9 + p * 9 + idx];
            Complex rise = val_im1 - val_im2;
            dip_guess[p * 9 + idx] = val_im1 + new_run * rise / run;
        }
    }
    return dip_guess;
}

// -----------------------------------------------------------------------------
// GMRES Solver Core (GPU-Resident)
// -----------------------------------------------------------------------------

std::vector<Complex> Dipole_Solver::compute_dipoles(const std::vector<Complex>& E, 
                                                    const std::vector<Complex>& dip_guess) {
    size_t vec_size = num_particles * 3;
    size_t restart = std::min(vec_size, static_cast<size_t>(10));
    size_t maxiter = std::min(vec_size, static_cast<size_t>(100));

    // 1. Prepare RHS b and copy to GPU
    std::vector<double> host_b(vec_size * 2, 0.0);
    for (size_t i = 0; i < vec_size; ++i) {
        host_b[i * 2 + 0] = E[i].real();
        host_b[i * 2 + 1] = E[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_b, host_b.data(), vec_size * 2 * sizeof(double), cudaMemcpyHostToDevice));

    double b_norm = gpu_norm(d_b, vec_size, d_reduce_buf);
    if (b_norm == 0.0) {
        return std::vector<Complex>(vec_size, 0.0);
    }

    // 2. Prepare initial guess x0 and copy to GPU
    std::vector<double> host_x(vec_size * 2, 0.0);
    for (size_t i = 0; i < vec_size; ++i) {
        host_x[i * 2 + 0] = dip_guess[i].real();
        host_x[i * 2 + 1] = dip_guess[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_x, host_x.data(), vec_size * 2 * sizeof(double), cudaMemcpyHostToDevice));

    // 3. Compute initial residual r = b - A(x)
    // Copy d_x to EF dipoles
    CUDA_CHECK(cudaMemcpy(EF->getDevDipoles(), d_x, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
    EF->calculate();
    // Copy computed E-field to d_r
    CUDA_CHECK(cudaMemcpy(d_r, EF->getDevEPoint(), vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
    // d_r = d_b - d_r
    gpu_vector_sub(d_r, d_b, d_r, vec_size);

    double r_norm = gpu_norm(d_r, vec_size, d_reduce_buf);
    if (r_norm / b_norm < tol) {
        std::vector<Complex> sol(vec_size);
        CUDA_CHECK(cudaMemcpy(host_x.data(), d_x, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < vec_size; ++i) {
            sol[i] = Complex(host_x[i * 2 + 0], host_x[i * 2 + 1]);
        }
        return sol;
    }

    // 4. Hessenberg and rotation buffers (Host-side)
    std::vector<std::vector<Complex>> H(restart + 1, std::vector<Complex>(restart, 0.0));
    std::vector<Complex> g(restart + 1, 0.0);
    std::vector<double> c(restart, 0.0);
    std::vector<Complex> s(restart, 0.0);

    size_t iter = 0;
    while (iter < maxiter) {
        // Compute current residual r = b - A(x)
        CUDA_CHECK(cudaMemcpy(EF->getDevDipoles(), d_x, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
        EF->calculate();
        CUDA_CHECK(cudaMemcpy(d_r, EF->getDevEPoint(), vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
        gpu_vector_sub(d_r, d_b, d_r, vec_size);
        
        r_norm = gpu_norm(d_r, vec_size, d_reduce_buf);
        if (r_norm / b_norm < tol) {
            break;
        }

        g[0] = r_norm;
        for (size_t i = 1; i <= restart; ++i) {
            g[i] = 0.0;
        }

        // V_0 = d_r / r_norm
        CUDA_CHECK(cudaMemcpy(d_V, d_r, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
        gpu_vector_scale(d_V, 1.0 / r_norm, vec_size);

        size_t k = 0;
        for (k = 0; k < restart && iter < maxiter; ++k, ++iter) {
            double* d_Vk = d_V + k * vec_size * 2;
            double* d_Vk1 = d_V + (k + 1) * vec_size * 2;

            // w = A(V_k)
            CUDA_CHECK(cudaMemcpy(EF->getDevDipoles(), d_Vk, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
            EF->calculate();
            CUDA_CHECK(cudaMemcpy(d_w, EF->getDevEPoint(), vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));

            // Arnoldi Gram-Schmidt orthogonalization
            for (size_t i = 0; i <= k; ++i) {
                double* d_Vi = d_V + i * vec_size * 2;
                H[i][k] = gpu_dot_product(d_Vi, d_w, vec_size, d_reduce_buf);
                gpu_vector_add(d_w, d_Vi, -H[i][k], vec_size);
            }

            double w_norm = gpu_norm(d_w, vec_size, d_reduce_buf);
            H[k + 1][k] = w_norm;

            if (w_norm > 0.0) {
                CUDA_CHECK(cudaMemcpy(d_Vk1, d_w, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
                gpu_vector_scale(d_Vk1, 1.0 / w_norm, vec_size);
            }

            // Apply previous Givens rotations to k-th column of H
            for (size_t i = 0; i < k; ++i) {
                Complex h1 = H[i][k];
                Complex h2 = H[i + 1][k];
                H[i][k] = c[i] * h1 + std::conj(s[i]) * h2;
                H[i + 1][k] = -s[i] * h1 + c[i] * h2;
            }

            // Determine next Givens rotation
            Complex h1 = H[k][k];
            Complex h2 = H[k + 1][k];
            double c_rot = 1.0;
            Complex s_rot = 0.0;
            if (h2 != 0.0) {
                double abs_h1 = std::abs(h1);
                double scale_val = abs_h1 + std::abs(h2);
                double norm_h1 = std::abs(h1 / scale_val);
                double norm_h2 = std::abs(h2 / scale_val);
                double h_hyp = scale_val * std::sqrt(norm_h1 * norm_h1 + norm_h2 * norm_h2);
                c_rot = abs_h1 / h_hyp;
                s_rot = (std::conj(h1) / abs_h1) * (h2 / h_hyp);
            }
            c[k] = c_rot;
            s[k] = s_rot;

            H[k][k] = c_rot * h1 + std::conj(s_rot) * h2;
            H[k + 1][k] = 0.0;

            // Apply new rotation to g
            Complex g1 = g[k];
            Complex g2 = g[k + 1];
            g[k] = c_rot * g1 + std::conj(s_rot) * g2;
            g[k + 1] = -s_rot * g1 + c_rot * g2;

            double resid = std::abs(g[k + 1]);
            if (resid / b_norm < tol) {
                k++; // Include the k-th component
                break;
            }
        }

        // Solve upper triangular Hessenberg system H y = g
        std::vector<Complex> y(k);
        for (int i = static_cast<int>(k) - 1; i >= 0; --i) {
            y[i] = g[i];
            for (size_t j = i + 1; j < k; ++j) {
                y[i] -= H[i][j] * y[j];
            }
            y[i] /= H[i][i];
        }

        // Update solution x = x + V * y
        for (size_t i = 0; i < k; ++i) {
            double* d_Vi = d_V + i * vec_size * 2;
            gpu_vector_add(d_x, d_Vi, y[i], vec_size);
        }
    }

    // Retrieve solution from GPU
    std::vector<Complex> sol(vec_size);
    CUDA_CHECK(cudaMemcpy(host_x.data(), d_x, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < vec_size; ++i) {
        sol[i] = Complex(host_x[i * 2 + 0], host_x[i * 2 + 1]);
    }
    return sol;
}

// -----------------------------------------------------------------------------
// Spectrum and Tensor computation loops
// -----------------------------------------------------------------------------

void Dipole_Solver::compute_tensor(const std::vector<Complex>& dip_guess, 
                                   std::vector<Complex>& frame_cap, 
                                   std::vector<Complex>& frame_dip) {
    size_t vec_size = num_particles * 3;
    
    // E represents unit fields for x, y, and z directions
    std::vector<std::vector<Complex>> E(3, std::vector<Complex>(vec_size, 0.0));
    for (int dim = 0; dim < 3; ++dim) {
        for (size_t p = 0; p < num_particles; ++p) {
            E[dim][p * 3 + dim] = 1.0;
        }
    }

    // Solve for x, y, and z orientations
    for (int dim = 0; dim < 3; ++dim) {
        std::vector<Complex> dip_guess_dim(vec_size);
        for (size_t p = 0; p < num_particles; ++p) {
            for (int c = 0; c < 3; ++c) {
                dip_guess_dim[p * 3 + c] = dip_guess[p * 9 + dim * 3 + c];
            }
        }

        std::vector<Complex> sol_dip = compute_dipoles(E[dim], dip_guess_dim);

        // Store solved dipoles
        for (size_t p = 0; p < num_particles; ++p) {
            for (int c = 0; c < 3; ++c) {
                frame_dip[p * 9 + dim * 3 + c] = sol_dip[p * 3 + c];
            }
        }

        // Average dipoles to get cap (tensor dimension dim)
        std::vector<Complex> avg_dip_dim(3, 0.0);
        for (size_t p = 0; p < num_particles; ++p) {
            avg_dip_dim[0] += sol_dip[p * 3 + 0];
            avg_dip_dim[1] += sol_dip[p * 3 + 1];
            avg_dip_dim[2] += sol_dip[p * 3 + 2];
        }
        for (int c = 0; c < 3; ++c) {
            frame_cap[dim * 3 + c] = avg_dip_dim[c] / static_cast<double>(num_particles);
        }
    }
}

std::vector<Complex> Dipole_Solver::compute_spectrum(const std::vector<Complex>& initial_guess) {
    // Returns full calculated dipoles for the frame of shape: num_wavevectors * num_particles * 9
    // And averages polarizabilities to frame_cap of shape: num_wavevectors * 9
    std::vector<Complex> frame_cap(num_wavevectors * 9, 0.0);
    std::vector<Complex> frame_dip(num_wavevectors * num_particles * 9, 0.0);

    std::vector<Complex> dip_guess = initial_guess;

    print("Wavenumber:");
    increase_indent();
    for (size_t wavevec_idx = 0; wavevec_idx < num_wavevectors; ++wavevec_idx) {
        std::string progress = std::to_string(wavevec_idx) + " of " + std::to_string(num_wavevectors);
        print(progress);

        // Calculate complex particle-dependent Ewald self coefficients
        std::vector<double> self_coef_r(num_particles);
        std::vector<double> self_coef_i(num_particles);
        for (size_t p = 0; p < num_particles; ++p) {
            Complex sc = -3.0 / (4.0 * M_PI * (1.0 - eps_p[wavevec_idx][p]));
            self_coef_r[p] = sc.real();
            self_coef_i[p] = sc.imag();
        }
        EF->setSelfCoef(self_coef_r, self_coef_i);

        // Calculate guess dipoles
        dip_guess = calc_guess(frame_dip, wavevec_idx);

        // Solve frame tensor
        std::vector<Complex> step_cap(9, 0.0);
        std::vector<Complex> step_dip(num_particles * 9, 0.0);
        compute_tensor(dip_guess, step_cap, step_dip);

        // Store wavevector results
        std::copy(step_cap.begin(), step_cap.end(), frame_cap.begin() + wavevec_idx * 9);
        std::copy(step_dip.begin(), step_dip.end(), frame_dip.begin() + wavevec_idx * num_particles * 9);
    }
    decrease_indent();

    // Store frame average polarizability in class avg_dips
    avg_dips.insert(avg_dips.end(), frame_cap.begin(), frame_cap.end());

    return frame_dip;
}

// -----------------------------------------------------------------------------
// Entry Point compute()
// -----------------------------------------------------------------------------

void Dipole_Solver::compute(const std::vector<double>& x_part,
                            const std::vector<double>& y_part,
                            const std::vector<double>& z_part) {
    if (x_part.size() != y_part.size() || x_part.size() != z_part.size()) {
        throw std::invalid_argument("Input coordinate vectors must have the exact same size.");
    }
    size_t num_p = x_part.size();
    if (num_p == 0) return;

    if (num_particles == 0) {
        set_dims(num_p);
        nondimensionalize();
        precalculations();
        allocate_gpu_buffers();

        // Instantiate CUDA-accelerated Electric_Field solver
        EF = std::make_unique<Electric_Field>(box[0], box[1], box[2], tol, xi, true);
    } else if (num_particles != num_p) {
        throw std::runtime_error("The number of particles has changed!");
    }

    num_frames++;
    if (num_frames != 1) {
        print("frame");
        increase_indent();
    }

    std::vector<double> scaled_x(num_particles);
    std::vector<double> scaled_y(num_particles);
    std::vector<double> scaled_z(num_particles);
    for (size_t i = 0; i < num_particles; ++i) {
        scaled_x[i] = x_part[i] / length_scale;
        scaled_y[i] = y_part[i] / length_scale;
        scaled_z[i] = z_part[i] / length_scale;
    }

    if (num_frames != 1) {
        std::string frame_msg = std::to_string(num_frames - 1) + " of " + std::to_string(num_frames);
        print(frame_msg);
        increase_indent();
    }

    // Set positions inside Electric_Field (also triggers internal coordinate updates and precalcs)
    EF->updateParticleCoordinates(scaled_x, scaled_y, scaled_z);

    // Solve spectrum
    std::vector<Complex> initial_guess = calc_mean_field_guess(0);
    std::vector<Complex> frame_dip = compute_spectrum(initial_guess);

    // Save solver dipoles in class dips
    dips.insert(dips.end(), frame_dip.begin(), frame_dip.end());

    if (num_frames != 1) {
        decrease_indent();
        decrease_indent();
    }
}

// -----------------------------------------------------------------------------
// Getters
// -----------------------------------------------------------------------------

std::vector<Complex> Dipole_Solver::get_eff_polarizability() const {
    // Average polarizabilities over all frames: shape: num_wavevectors * 9
    std::vector<Complex> eff_polarizability(num_wavevectors * 9, 0.0);
    if (num_frames == 0) return eff_polarizability;

    for (size_t w = 0; w < num_wavevectors; ++w) {
        for (int idx = 0; idx < 9; ++idx) {
            Complex sum = 0.0;
            for (size_t f = 0; f < num_frames; ++f) {
                sum += avg_dips[f * num_wavevectors * 9 + w * 9 + idx];
            }
            eff_polarizability[w * 9 + idx] = sum / static_cast<double>(num_frames);
        }
    }
    return eff_polarizability;
}

std::vector<Complex> Dipole_Solver::get_dipoles() const {
    return dips;
}
