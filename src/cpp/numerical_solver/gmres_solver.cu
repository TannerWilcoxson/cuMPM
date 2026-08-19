#include "gmres_solver.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

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
// GMRES_Solver Class Implementation
// -----------------------------------------------------------------------------

GMRES_Solver::~GMRES_Solver() {
    free_buffers();
}

void GMRES_Solver::initialize(size_t vec_size) {
    if (vec_size == 0) return;
    if (vec_size == allocated_vec_size && d_x != nullptr) return;

    free_buffers();

    size_t size_bytes = vec_size * sizeof(Complex);
    size_t active_restart = std::min(vec_size, this->restart);

    CUDA_CHECK(cudaMalloc(&d_x, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_r, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_w, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, (active_restart + 1) * size_bytes));
    CUDA_CHECK(cudaMalloc(&d_reduce_buf, 2 * sizeof(double)));

    allocated_vec_size = vec_size;
}

void GMRES_Solver::free_buffers() {
    if (d_x) { cudaFree(d_x); d_x = nullptr; }
    if (d_b) { cudaFree(d_b); d_b = nullptr; }
    if (d_r) { cudaFree(d_r); d_r = nullptr; }
    if (d_w) { cudaFree(d_w); d_w = nullptr; }
    if (d_V) { cudaFree(d_V); d_V = nullptr; }
    if (d_reduce_buf) { cudaFree(d_reduce_buf); d_reduce_buf = nullptr; }
    allocated_vec_size = 0;
}

std::vector<Complex> GMRES_Solver::solve(
    const std::vector<Complex>& b,
    const std::vector<Complex>& x0,
    Electric_Field* EF,
    double tol,
    bool quiet
) {
    size_t vec_size = b.size();
    if (vec_size == 0) return std::vector<Complex>();

    initialize(vec_size);

    size_t active_restart = std::min(vec_size, this->restart);
    size_t active_maxiter = this->maxiter;

    // 1. Copy RHS b directly to GPU
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), vec_size * sizeof(Complex), cudaMemcpyHostToDevice));

    double b_norm = gpu_norm(d_b, vec_size, d_reduce_buf);
    if (b_norm == 0.0) {
        return std::vector<Complex>(vec_size, 0.0);
    }

    // 2. Copy initial guess x0 directly to GPU
    CUDA_CHECK(cudaMemcpy(d_x, x0.data(), vec_size * sizeof(Complex), cudaMemcpyHostToDevice));

    // 3. Hessenberg and rotation buffers (Host-side)
    std::vector<std::vector<Complex>> H(active_restart + 1, std::vector<Complex>(active_restart, 0.0));
    std::vector<Complex> g(active_restart + 1, 0.0);
    std::vector<double> c(active_restart, 0.0);
    std::vector<Complex> s(active_restart, 0.0);

    size_t iter = 0;
    while (iter < active_maxiter) {
        // Compute current residual r = b - A(x)
        compute_Ax(d_x, d_r, EF, vec_size);
        // d_r = d_b - d_r
        gpu_vector_sub(d_r, d_b, d_r, vec_size);
        if (use_jacobi_precond) {
            gpu_vector_jacobi_precond(d_w, d_r, EF, vec_size);
            CUDA_CHECK(cudaMemcpy(d_r, d_w, vec_size * sizeof(Complex), cudaMemcpyDeviceToDevice));
        }

        double r_norm = gpu_norm(d_r, vec_size, d_reduce_buf);
        if (r_norm < tol * b_norm) {
            break;
        }

        g[0] = r_norm;
        for (size_t i = 1; i <= active_restart; ++i) {
            g[i] = 0.0;
        }

        // V_0 = d_r / r_norm
        CUDA_CHECK(cudaMemcpy(d_V, d_r, vec_size * sizeof(Complex), cudaMemcpyDeviceToDevice));
        gpu_vector_scale(d_V, 1.0 / r_norm, vec_size);

        size_t k = 0;
        for (k = 0; k < active_restart && iter < active_maxiter; ++k, ++iter) {
            double* d_Vk = d_V + k * vec_size * 2;
            double* d_Vk1 = d_V + (k + 1) * vec_size * 2;

            // w = A(V_k)
            compute_Ax(d_Vk, d_w, EF, vec_size);
            if (use_jacobi_precond) {
                gpu_vector_jacobi_precond(d_r, d_w, EF, vec_size);
                CUDA_CHECK(cudaMemcpy(d_w, d_r, vec_size * sizeof(Complex), cudaMemcpyDeviceToDevice));
            }

            // Arnoldi Gram-Schmidt orthogonalization
            for (size_t i = 0; i <= k; ++i) {
                double* d_Vi = d_V + i * vec_size * 2;
                H[i][k] = gpu_dot_product(d_Vi, d_w, vec_size, d_reduce_buf);
                gpu_vector_add(d_w, d_Vi, -H[i][k], vec_size);
            }

            double w_norm = gpu_norm(d_w, vec_size, d_reduce_buf);
            H[k + 1][k] = w_norm;

            if (w_norm > 0.0) {
                CUDA_CHECK(cudaMemcpy(d_Vk1, d_w, vec_size * sizeof(Complex), cudaMemcpyDeviceToDevice));
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
                if (abs_h1 == 0.0) {
                    c_rot = 0.0;
                    s_rot = 1.0;
                } else {
                    double scale_val = abs_h1 + std::abs(h2);
                    double norm_h1 = std::abs(h1 / scale_val);
                    double norm_h2 = std::abs(h2 / scale_val);
                    double h_hyp = scale_val * std::sqrt(norm_h1 * norm_h1 + norm_h2 * norm_h2);
                    c_rot = abs_h1 / h_hyp;
                    s_rot = (std::conj(h1) / abs_h1) * (h2 / h_hyp);
                }
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
            if (resid < tol * b_norm) {
                k++; // Include the k-th component
                iter++; // Skip remainder of step count for this block
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
    CUDA_CHECK(cudaMemcpy(sol.data(), d_x, vec_size * sizeof(Complex), cudaMemcpyDeviceToHost));
    return sol;
}
