#include "numerical_solver.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

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

__global__ void dot_product_unconjugated_kernel(const double* a, const double* b, double* out_r, double* out_i, size_t size) {
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

        // Unconjugated multiplication: (ar + i*ai)*(br + i*bi)
        sum_r = ar * br - ai * bi;
        sum_i = ar * bi + ai * br;
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
// Numerical_Solver Base Class Helper Implementations
// -----------------------------------------------------------------------------

void Numerical_Solver::gpu_vector_add(double* d_y, const double* d_x, Complex alpha, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(d_y, d_x, alpha.real(), alpha.imag(), size);
    CUDA_CHECK(cudaGetLastError());
}

void Numerical_Solver::gpu_vector_scale(double* d_y, Complex alpha, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    vector_scale_kernel<<<blocks, threads>>>(d_y, alpha.real(), alpha.imag(), size);
    CUDA_CHECK(cudaGetLastError());
}

void Numerical_Solver::gpu_vector_sub(double* d_y, const double* d_x, const double* d_z, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    vector_sub_kernel<<<blocks, threads>>>(d_y, d_x, d_z, size);
    CUDA_CHECK(cudaGetLastError());
}

Complex Numerical_Solver::gpu_dot_product(const double* d_a, const double* d_b, size_t size, double* d_reduce_buf) {
    CUDA_CHECK(cudaMemset(d_reduce_buf, 0, 2 * sizeof(double)));
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dot_product_kernel<<<blocks, threads>>>(d_a, d_b, &d_reduce_buf[0], &d_reduce_buf[1], size);
    CUDA_CHECK(cudaGetLastError());

    double host_res[2];
    CUDA_CHECK(cudaMemcpy(host_res, d_reduce_buf, 2 * sizeof(double), cudaMemcpyDeviceToHost));
    return Complex(host_res[0], host_res[1]);
}

Complex Numerical_Solver::gpu_dot_product_unconjugated(const double* d_a, const double* d_b, size_t size, double* d_reduce_buf) {
    CUDA_CHECK(cudaMemset(d_reduce_buf, 0, 2 * sizeof(double)));
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dot_product_unconjugated_kernel<<<blocks, threads>>>(d_a, d_b, &d_reduce_buf[0], &d_reduce_buf[1], size);
    CUDA_CHECK(cudaGetLastError());

    double host_res[2];
    CUDA_CHECK(cudaMemcpy(host_res, d_reduce_buf, 2 * sizeof(double), cudaMemcpyDeviceToHost));
    return Complex(host_res[0], host_res[1]);
}

double Numerical_Solver::gpu_norm(const double* d_a, size_t size, double* d_reduce_buf) {
    CUDA_CHECK(cudaMemset(d_reduce_buf, 0, sizeof(double)));
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    norm_kernel<<<blocks, threads>>>(d_a, d_reduce_buf, size);
    CUDA_CHECK(cudaGetLastError());

    double host_res = 0.0;
    CUDA_CHECK(cudaMemcpy(&host_res, d_reduce_buf, sizeof(double), cudaMemcpyDeviceToHost));
    return std::sqrt(host_res);
}

__global__ void jacobi_precond_kernel(
    double* __restrict__ d_dst,
    const double* __restrict__ d_src,
    const double* __restrict__ d_self_r,
    const double* __restrict__ d_self_i,
    size_t num_particles,
    bool solve_quadrupoles,
    size_t num_quads,
    const int* __restrict__ d_quad_map)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_particles) {
        double sr = d_self_r ? d_self_r[idx] : 1.0;
        double si = d_self_i ? d_self_i[idx] : 0.0;
        double denom = sr * sr + si * si;
        double inv_sr = (denom > 1e-30) ? (sr / denom) : 1.0;
        double inv_si = (denom > 1e-30) ? (-si / denom) : 0.0;

        for (int c = 0; c < 3; ++c) {
            size_t comp_idx = (idx * 3 + c) * 2;
            double xr = d_src[comp_idx + 0];
            double xi = d_src[comp_idx + 1];

            d_dst[comp_idx + 0] = xr * inv_sr - xi * inv_si;
            d_dst[comp_idx + 1] = xr * inv_si + xi * inv_sr;
        }
    }

    if (solve_quadrupoles && idx < num_quads) {
        size_t p_idx = d_quad_map ? d_quad_map[idx] : idx;
        if (p_idx >= num_particles) p_idx = 0;
        double sr = d_self_r ? d_self_r[p_idx] : 1.0;
        double si = d_self_i ? d_self_i[p_idx] : 0.0;
        double denom = sr * sr + si * si;
        double inv_sr = (denom > 1e-30) ? (sr / denom) : 1.0;
        double inv_si = (denom > 1e-30) ? (-si / denom) : 0.0;

        size_t quad_offset = num_particles * 3 * 2;
        for (int c = 0; c < 5; ++c) {
            size_t comp_idx = quad_offset + (idx * 5 + c) * 2;
            double xr = d_src[comp_idx + 0];
            double xi = d_src[comp_idx + 1];

            d_dst[comp_idx + 0] = xr * inv_sr - xi * inv_si;
            d_dst[comp_idx + 1] = xr * inv_si + xi * inv_sr;
        }
    }
}

void Numerical_Solver::gpu_vector_jacobi_precond(double* d_dst, const double* d_src, const Electric_Field* EF, size_t vec_size) {
    const Base_Electric_Field* base_ef = dynamic_cast<const Base_Electric_Field*>(EF);
    if (!base_ef || !base_ef->getDevSelfCoefReal()) {
        CUDA_CHECK(cudaMemcpy(d_dst, d_src, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
        return;
    }

    size_t num_particles = base_ef->getNumParticles();
    bool solve_quads = base_ef->getSolveQuadrupoles();
    size_t num_quads = base_ef->getNumQuads();
    const int* d_quad_map = base_ef->getDevQuadMap();

    const double* d_self_r = base_ef->getDevSelfCoefReal();
    const double* d_self_i = base_ef->getDevSelfCoefImag();

    size_t max_items = num_particles;
    if (solve_quads && num_quads > max_items) max_items = num_quads;

    if (max_items == 0) {
        CUDA_CHECK(cudaMemcpy(d_dst, d_src, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
        return;
    }

    int threads = 256;
    int blocks = (max_items + threads - 1) / threads;
    jacobi_precond_kernel<<<blocks, threads>>>(d_dst, d_src, d_self_r, d_self_i, num_particles, solve_quads, num_quads, d_quad_map);
    CUDA_CHECK(cudaGetLastError());
}

void Numerical_Solver::compute_Ax(double* d_x, double* d_Ax, Electric_Field* EF, size_t vec_size) {
    CUDA_CHECK(cudaMemcpy(EF->getDevDipoles(), d_x, vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
    EF->calculate();
    CUDA_CHECK(cudaMemcpy(d_Ax, EF->getDevEPoint(), vec_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
}

void Numerical_Solver::compute_Ax_preconditioned(double* d_x, double* d_Ax, double* d_tmp, Electric_Field* EF, size_t vec_size) {
    gpu_vector_jacobi_precond(d_tmp, d_x, EF, vec_size);
    compute_Ax(d_tmp, d_Ax, EF, vec_size);
}

