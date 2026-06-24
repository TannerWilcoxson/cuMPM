#ifndef BICGSTAB_SOLVER_H
#define BICGSTAB_SOLVER_H

#include "numerical_solver.h"

class BiCGSTAB_Solver : public Numerical_Solver {
private:
    double* d_x = nullptr;
    double* d_b = nullptr;
    double* d_r0tilde = nullptr;
    double* d_r = nullptr;
    double* d_p = nullptr;
    double* d_v = nullptr;
    double* d_s = nullptr;
    double* d_t = nullptr;
    double* d_tmp = nullptr;
    double* d_reduce_buf = nullptr;
    size_t allocated_vec_size = 0;

    void free_buffers();

public:
    BiCGSTAB_Solver() = default;
    ~BiCGSTAB_Solver() override;

    BiCGSTAB_Solver(const BiCGSTAB_Solver&) = delete;
    BiCGSTAB_Solver& operator=(const BiCGSTAB_Solver&) = delete;

    void initialize(size_t vec_size) override;

    std::vector<Complex> solve(
        const std::vector<Complex>& b,
        const std::vector<Complex>& x0,
        Electric_Field* EF,
        double tol
    ) override;
};

#endif // BICGSTAB_SOLVER_H
