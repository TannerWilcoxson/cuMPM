#ifndef NUMERICAL_SOLVER_H
#define NUMERICAL_SOLVER_H

#include <vector>
#include <complex>
#include "electric_field/electric_field.h"

using Complex = std::complex<double>;

class Numerical_Solver {
protected:
    // Shared GPU vector operation helpers (implemented in numerical_solver.cu)
    void gpu_vector_add(double2* d_y, const double2* d_x, Complex alpha, size_t size);
    void gpu_vector_scale(double2* d_y, Complex alpha, size_t size);
    void gpu_vector_sub(double2* d_y, const double2* d_x, const double2* d_z, size_t size);
    Complex gpu_dot_product(const double2* d_a, const double2* d_b, size_t size, double* d_reduce_buf);
    Complex gpu_dot_product_unconjugated(const double2* d_a, const double2* d_b, size_t size, double* d_reduce_buf);
    double gpu_norm(const double2* d_a, size_t size, double* d_reduce_buf);
    void gpu_vector_jacobi_precond(double2* d_dst, const double2* d_src, const Electric_Field* EF, size_t vec_size);
    void compute_Ax(double2* d_x, double2* d_Ax, Electric_Field* EF, size_t vec_size);
    void compute_Ax_preconditioned(double2* d_x, double2* d_Ax, double2* d_tmp, Electric_Field* EF, size_t vec_size);

protected:
    bool use_jacobi_precond = true;

public:
    virtual ~Numerical_Solver() = default;

    bool isUsingJacobiPrecond() const { return use_jacobi_precond; }
    void setUseJacobiPrecond(bool enable) { use_jacobi_precond = enable; }

    virtual void initialize(size_t vec_size) {}

    virtual std::vector<Complex> solve(
        const std::vector<Complex>& b,
        const std::vector<Complex>& x0,
        Electric_Field* EF,
        double tol,
        bool quiet = false
    ) = 0;
};

#endif // NUMERICAL_SOLVER_H
