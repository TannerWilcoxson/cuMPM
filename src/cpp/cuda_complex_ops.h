#ifndef CUDA_COMPLEX_OPS_H
#define CUDA_COMPLEX_OPS_H

#include <cuda_runtime.h>
#include <type_traits>

// -----------------------------------------------------------------------------
// Type Traits for CUDA 2D Complex Vectors (double2, float2)
// -----------------------------------------------------------------------------

template <typename T>
struct is_cuda_vec2 : std::false_type {};

template <> struct is_cuda_vec2<double2> : std::true_type {};
template <> struct is_cuda_vec2<float2>  : std::true_type {};

template <typename Real> struct Real2Traits;

template <> struct Real2Traits<double> {
    using Vec2 = double2;
    __host__ __device__ static inline double2 make(double x, double y) {
        return make_double2(x, y);
    }
};

template <> struct Real2Traits<float> {
    using Vec2 = float2;
    __host__ __device__ static inline float2 make(float x, float y) {
        return make_float2(x, y);
    }
};

// -----------------------------------------------------------------------------
// Templated Complex Arithmetic Operators for CUDA Vec2 (double2, float2)
// -----------------------------------------------------------------------------

template <typename Vec2, typename = std::enable_if_t<is_cuda_vec2<Vec2>::value>>
__host__ __device__ inline Vec2 operator+(const Vec2& a, const Vec2& b) {
    Vec2 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

template <typename Vec2, typename = std::enable_if_t<is_cuda_vec2<Vec2>::value>>
__host__ __device__ inline Vec2 operator-(const Vec2& a, const Vec2& b) {
    Vec2 r;
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    return r;
}

template <typename Vec2, typename = std::enable_if_t<is_cuda_vec2<Vec2>::value>>
__host__ __device__ inline Vec2 operator*(const Vec2& a, const Vec2& b) {
    Vec2 r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}

template <typename Vec2, typename S, typename = std::enable_if_t<is_cuda_vec2<Vec2>::value && std::is_arithmetic<S>::value>>
__host__ __device__ inline Vec2 operator*(S s, const Vec2& a) {
    Vec2 r;
    using Scalar = decltype(a.x);
    r.x = static_cast<Scalar>(s) * a.x;
    r.y = static_cast<Scalar>(s) * a.y;
    return r;
}

template <typename Vec2, typename S, typename = std::enable_if_t<is_cuda_vec2<Vec2>::value && std::is_arithmetic<S>::value>>
__host__ __device__ inline Vec2 operator*(const Vec2& a, S s) {
    return s * a;
}

template <typename Vec2, typename = std::enable_if_t<is_cuda_vec2<Vec2>::value>>
__host__ __device__ inline Vec2 conj(const Vec2& a) {
    Vec2 r;
    r.x = a.x;
    r.y = -a.y;
    return r;
}

template <typename Vec2, typename = std::enable_if_t<is_cuda_vec2<Vec2>::value>>
__host__ __device__ inline Vec2& operator+=(Vec2& a, const Vec2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

template <typename Vec2, typename = std::enable_if_t<is_cuda_vec2<Vec2>::value>>
__host__ __device__ inline Vec2& operator-=(Vec2& a, const Vec2& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

__global__ inline void cast_double_to_float_kernel(const double* __restrict__ src, float* __restrict__ dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

#endif // CUDA_COMPLEX_OPS_H
