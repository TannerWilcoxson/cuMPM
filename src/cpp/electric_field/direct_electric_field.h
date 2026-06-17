#ifndef DIRECT_ELECTRIC_FIELD_H
#define DIRECT_ELECTRIC_FIELD_H

#include <vector>
#include "electric_field.h"

/**
 * @class Direct_Electric_Field
 * @brief Computes dipole-dipole interactions using the free-space (open boundary)
 *        quasi-static Green's function via an O(N^2) all-pairs CUDA kernel.
 *
 * Unlike the Ewald-based solvers, this class does not impose periodic boundary
 * conditions. It is appropriate for isolated clusters (EELS, single-particle
 * near-field), and serves as a reference implementation for validating the
 * Ewald solvers in the large-box limit.
 *
 * The interaction tensor applied to source dipole p at position r_i for the
 * field at r_j (i != j) is the quasi-static dipole-dipole tensor:
 *
 *   T_ij * p = (3*(p.r_hat)*r_hat - p) / r^3      (in Gaussian-like units)
 *
 * The self-interaction coefficient (self_coef_r, self_coef_i per particle) is
 * supplied externally by the Dipole_Solver and matches the polarizability
 * definition used by the Ewald solvers.
 */
class Direct_Electric_Field : public Electric_Field {
private:
    size_t num_particles = 0;

    double* d_x_part = nullptr;
    double* d_y_part = nullptr;
    double* d_z_part = nullptr;

    // Complex dipole source array: layout (N*3) double2 values, matching Ewald convention
    double* d_dipoles = nullptr;

    // Complex output field at each particle: same layout as d_dipoles
    double* d_E_point = nullptr;

    // Per-particle self-interaction coefficients (real and imaginary parts)
    double* d_self_coef_r = nullptr;
    double* d_self_coef_i = nullptr;

    std::vector<double> host_radius;
    double* d_radius = nullptr;

    bool particles_updated = false;

    void electricField();

public:
    Direct_Electric_Field(const std::vector<double>& radius = {});
    ~Direct_Electric_Field() override;

    Direct_Electric_Field(const Direct_Electric_Field&) = delete;
    Direct_Electric_Field& operator=(const Direct_Electric_Field&) = delete;

    double* getDevDipoles() const override { return d_dipoles; }
    double* getDevEPoint()  const override { return d_E_point; }

    void calculate() override;

    void updateParticleCoordinates(const std::vector<double>& x_part,
                                   const std::vector<double>& y_part,
                                   const std::vector<double>& z_part) override;

    void setSelfCoef(const std::vector<double>& self_coef_r,
                     const std::vector<double>& self_coef_i) override;
};

#endif // DIRECT_ELECTRIC_FIELD_H
