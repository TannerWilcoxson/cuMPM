#ifndef NUMERICAL_SOLVER_H
#define NUMERICAL_SOLVER_H

#include <vector>
#include <complex>
#include "electric_field/electric_field.h"

using Complex = std::complex<double>;

class Numerical_Solver {
protected:
    // Shared GPU vector operation helpers (implemented in numerical_solver.cu)
    void gpu_vector_add(double* d_y, const double* d_x, Complex alpha, size_t size);
    void gpu_vector_scale(double* d_y, Complex alpha, size_t size);
    void gpu_vector_sub(double* d_y, const double* d_x, const double* d_z, size_t size);
    Complex gpu_dot_product(const double* d_a, const double* d_b, size_t size, double* d_reduce_buf);
    double gpu_norm(const double* d_a, size_t size, double* d_reduce_buf);
    void compute_Ax(double* d_x, double* d_Ax, Electric_Field* EF, size_t vec_size);

public:
    virtual ~Numerical_Solver() = default;

    virtual void initialize(size_t num_particles) {}

    virtual std::vector<Complex> solve(
        const std::vector<Complex>& b,
        const std::vector<Complex>& x0,
        Electric_Field* EF,
        double tol
    ) = 0;
};

#endif // NUMERICAL_SOLVER_H
