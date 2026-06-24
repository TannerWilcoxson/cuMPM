#include "bicgstab_solver.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#ifndef CUDA_CHECK
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
      if (abort) exit(code);
   }
}
#endif

BiCGSTAB_Solver::~BiCGSTAB_Solver() {
    free_buffers();
}

void BiCGSTAB_Solver::initialize(size_t vec_size) {
    if (vec_size == 0) return;
    if (vec_size == allocated_vec_size && d_x != nullptr) return;

    free_buffers();

    size_t size_bytes = vec_size * 2 * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_x, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_r0tilde, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_r, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_p, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_v, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_s, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_t, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_reduce_buf, 2 * sizeof(double)));

    allocated_vec_size = vec_size;
}

void BiCGSTAB_Solver::free_buffers() {
    if (d_x) { cudaFree(d_x); d_x = nullptr; }
    if (d_b) { cudaFree(d_b); d_b = nullptr; }
    if (d_r0tilde) { cudaFree(d_r0tilde); d_r0tilde = nullptr; }
    if (d_r) { cudaFree(d_r); d_r = nullptr; }
    if (d_p) { cudaFree(d_p); d_p = nullptr; }
    if (d_v) { cudaFree(d_v); d_v = nullptr; }
    if (d_s) { cudaFree(d_s); d_s = nullptr; }
    if (d_t) { cudaFree(d_t); d_t = nullptr; }
    if (d_tmp) { cudaFree(d_tmp); d_tmp = nullptr; }
    if (d_reduce_buf) { cudaFree(d_reduce_buf); d_reduce_buf = nullptr; }
    allocated_vec_size = 0;
}

std::vector<Complex> BiCGSTAB_Solver::solve(
    const std::vector<Complex>& b,
    const std::vector<Complex>& x0,
    Electric_Field* EF,
    double tol
) {
    size_t vec_size = b.size();
    if (vec_size == 0) return std::vector<Complex>();

    initialize(vec_size);

    size_t maxiter = std::min(vec_size, static_cast<size_t>(100));
    size_t size_bytes = vec_size * 2 * sizeof(double);

    // 1. Copy RHS b to GPU
    std::vector<double> host_b(vec_size * 2, 0.0);
    for (size_t i = 0; i < vec_size; ++i) {
        host_b[i * 2 + 0] = b[i].real();
        host_b[i * 2 + 1] = b[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_b, host_b.data(), size_bytes, cudaMemcpyHostToDevice));

    double b_norm = gpu_norm(d_b, vec_size, d_reduce_buf);
    if (b_norm == 0.0) {
        return std::vector<Complex>(vec_size, 0.0);
    }

    // 2. Copy initial guess x0 to GPU
    std::vector<double> host_x(vec_size * 2, 0.0);
    for (size_t i = 0; i < vec_size; ++i) {
        host_x[i * 2 + 0] = x0[i].real();
        host_x[i * 2 + 1] = x0[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_x, host_x.data(), size_bytes, cudaMemcpyHostToDevice));

    // 3. Compute initial residual r = b - A(x0)
    compute_Ax(d_x, d_tmp, EF, vec_size);
    gpu_vector_sub(d_r, d_b, d_tmp, vec_size);

    double r_norm = gpu_norm(d_r, vec_size, d_reduce_buf);
    if (r_norm / b_norm < tol) {
        std::vector<Complex> sol(vec_size);
        CUDA_CHECK(cudaMemcpy(host_x.data(), d_x, size_bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < vec_size; ++i) {
            sol[i] = Complex(host_x[i * 2 + 0], host_x[i * 2 + 1]);
        }
        return sol;
    }

    // 4. Choose shadow residual r0tilde = r0
    CUDA_CHECK(cudaMemcpy(d_r0tilde, d_r, size_bytes, cudaMemcpyDeviceToDevice));

    // 5. Initialize: p0 = r0, rho0 = 1, alpha = 1, omega = 1, v0 = 0
    CUDA_CHECK(cudaMemcpy(d_p, d_r, size_bytes, cudaMemcpyDeviceToDevice));
    Complex rho = 1.0;
    Complex alpha = 1.0;
    Complex omega = 1.0;

    CUDA_CHECK(cudaMemset(d_v, 0, size_bytes));

    for (size_t iter = 0; iter < maxiter; ++iter) {
        Complex rho_new = gpu_dot_product(d_r0tilde, d_r, vec_size, d_reduce_buf);
        if (std::abs(rho_new) == 0.0) {
            break; // Method failed
        }

        if (iter > 0) {
            Complex beta = (rho_new / rho) * (alpha / omega);
            // p = r + beta * (p - omega * v)
            // 1. p = p - omega * v
            gpu_vector_add(d_p, d_v, -omega, vec_size);
            // 2. p = beta * p
            gpu_vector_scale(d_p, beta, vec_size);
            // 3. p = p + r (which is p = p + 1.0 * r)
            gpu_vector_add(d_p, d_r, 1.0, vec_size);
        }

        rho = rho_new;

        // v = A(p)
        compute_Ax(d_p, d_v, EF, vec_size);

        Complex r0tilde_dot_v = gpu_dot_product(d_r0tilde, d_v, vec_size, d_reduce_buf);
        if (std::abs(r0tilde_dot_v) == 0.0) {
            break; // Stagnation
        }
        alpha = rho / r0tilde_dot_v;

        // s = r - alpha * v
        CUDA_CHECK(cudaMemcpy(d_s, d_r, size_bytes, cudaMemcpyDeviceToDevice));
        gpu_vector_add(d_s, d_v, -alpha, vec_size);

        // Check norm of s
        double s_norm = gpu_norm(d_s, vec_size, d_reduce_buf);
        if (s_norm / b_norm < tol) {
            // x = x + alpha * p
            gpu_vector_add(d_x, d_p, alpha, vec_size);
            break;
        }

        // t = A(s)
        compute_Ax(d_s, d_t, EF, vec_size);

        // omega = dot(t, s) / dot(t, t)
        Complex t_dot_s = gpu_dot_product(d_t, d_s, vec_size, d_reduce_buf);
        Complex t_dot_t = gpu_dot_product(d_t, d_t, vec_size, d_reduce_buf);
        if (std::abs(t_dot_t) == 0.0) {
            break;
        }
        omega = t_dot_s / t_dot_t.real();

        // x = x + alpha * p + omega * s
        gpu_vector_add(d_x, d_p, alpha, vec_size);
        gpu_vector_add(d_x, d_s, omega, vec_size);

        // r = s - omega * t
        CUDA_CHECK(cudaMemcpy(d_r, d_s, size_bytes, cudaMemcpyDeviceToDevice));
        gpu_vector_add(d_r, d_t, -omega, vec_size);

        // Check convergence on r
        r_norm = gpu_norm(d_r, vec_size, d_reduce_buf);
        if (r_norm / b_norm < tol) {
            break;
        }

        if (std::abs(omega) == 0.0) {
            break; // Stagnation
        }
    }

    // Retrieve solution from GPU
    std::vector<Complex> sol(vec_size);
    CUDA_CHECK(cudaMemcpy(host_x.data(), d_x, size_bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < vec_size; ++i) {
        sol[i] = Complex(host_x[i * 2 + 0], host_x[i * 2 + 1]);
    }
    return sol;
}
