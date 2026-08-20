#ifndef BICGSTAB_SOLVER_H
#define BICGSTAB_SOLVER_H

#include "numerical_solver.h"

class BiCGSTAB_Solver : public Numerical_Solver {
private:
    double2* d_x = nullptr;
    double2* d_b = nullptr;
    double2* d_r0tilde = nullptr;
    double2* d_r = nullptr;
    double2* d_p = nullptr;
    double2* d_v = nullptr;
    double2* d_s = nullptr;
    double2* d_t = nullptr;
    double2* d_tmp = nullptr;
    double* d_reduce_buf = nullptr;
    size_t allocated_vec_size = 0;
    size_t maxiter = 1000;

    void free_buffers();

public:
    BiCGSTAB_Solver(size_t maxiter = 1000)
        : maxiter(maxiter) {}
    ~BiCGSTAB_Solver() override;

    BiCGSTAB_Solver(const BiCGSTAB_Solver&) = delete;
    BiCGSTAB_Solver& operator=(const BiCGSTAB_Solver&) = delete;

    void initialize(size_t vec_size) override;

    std::vector<Complex> solve(
        const std::vector<Complex>& b,
        const std::vector<Complex>& x0,
        Electric_Field* EF,
        double tol,
        bool quiet = false
    ) override;
};

#endif // BICGSTAB_SOLVER_H
