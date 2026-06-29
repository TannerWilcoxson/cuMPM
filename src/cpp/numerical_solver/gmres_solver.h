#ifndef GMRES_SOLVER_H
#define GMRES_SOLVER_H

#include "numerical_solver.h"

class GMRES_Solver : public Numerical_Solver {
private:
    double* d_x = nullptr;
    double* d_b = nullptr;
    double* d_r = nullptr;
    double* d_w = nullptr;
    double* d_V = nullptr;
    double* d_reduce_buf = nullptr;
    size_t allocated_vec_size = 0;
    size_t restart = 50;
    size_t maxiter = 1000;

    void free_buffers();

public:
    GMRES_Solver(size_t restart = 50, size_t maxiter = 1000)
        : restart(restart), maxiter(maxiter) {}
    ~GMRES_Solver() override;

    GMRES_Solver(const GMRES_Solver&) = delete;
    GMRES_Solver& operator=(const GMRES_Solver&) = delete;

    void initialize(size_t vec_size) override;

    std::vector<Complex> solve(
        const std::vector<Complex>& b,
        const std::vector<Complex>& x0,
        Electric_Field* EF,
        double tol,
        bool quiet = false
    ) override;
};

#endif // GMRES_SOLVER_H
