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

    void free_buffers();

public:
    GMRES_Solver() = default;
    ~GMRES_Solver() override;

    GMRES_Solver(const GMRES_Solver&) = delete;
    GMRES_Solver& operator=(const GMRES_Solver&) = delete;

    void initialize(size_t vec_size) override;

    std::vector<Complex> solve(
        const std::vector<Complex>& b,
        const std::vector<Complex>& x0,
        Electric_Field* EF,
        double tol
    ) override;
};

#endif // GMRES_SOLVER_H
