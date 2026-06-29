#ifndef DIRECT_ELECTRIC_FIELD_H
#define DIRECT_ELECTRIC_FIELD_H

#include <vector>
#include <complex>
#include "electric_field.h"

using Complex = std::complex<double>;

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

    double* d_x_field = nullptr;
    double* d_y_field = nullptr;
    double* d_z_field = nullptr;
    size_t num_field_points = 0;
    bool field_points_updated = false;

    std::vector<double> host_radius;
    double* d_radius = nullptr;

    bool particles_updated = false;
    bool dipoles_updated = false;

    // Quadrupole member variables
    bool solve_quadrupoles = false;
    std::vector<int> quad_idxs;
    int* d_quad_idxs = nullptr;
    int* d_quad_map = nullptr;
    size_t num_quads = 0;
    FieldCalcMode mode = FieldCalcMode::SOLVER_AX;

    void electricField();

public:
    Direct_Electric_Field(const std::vector<double>& radius = {},
                          FieldCalcMode mode = FieldCalcMode::SOLVER_AX,
                          bool solve_quadrupoles = false,
                          const std::vector<int>& quad_idxs = {});
    FieldCalcMode getCalcMode() const { return mode; }
    ~Direct_Electric_Field() override;

    Direct_Electric_Field(const Direct_Electric_Field&) = delete;
    Direct_Electric_Field& operator=(const Direct_Electric_Field&) = delete;

    double* getDevDipoles() const override { return d_dipoles; }
    double* getDevEPoint()  const override { return d_E_point; }
    bool getSolveQuadrupoles() const { return solve_quadrupoles; }
    size_t getNumQuads() const { return num_quads; }
    size_t getNumFieldPoints() const { return num_field_points; }
    size_t getNumParticles() const { return num_particles; }
    std::vector<Complex> getEPointHost() const;

    bool getParticlesUpdated() const { return particles_updated; }
    bool getFieldPointsUpdated() const { return field_points_updated; }
    void clearParticlesUpdated() { particles_updated = false; }
    void clearFieldPointsUpdated() { field_points_updated = false; }

    bool getDipolesUpdated() const { return dipoles_updated; }
    void clearDipolesUpdated() { dipoles_updated = false; }

    double* getDevXPart() const { return d_x_part; }
    double* getDevYPart() const { return d_y_part; }
    double* getDevZPart() const { return d_z_part; }
    double* getDevXField() const { return d_x_field; }
    double* getDevYField() const { return d_y_field; }
    double* getDevZField() const { return d_z_field; }
    double* getDevSelfCoefReal() const { return d_self_coef_r; }
    double* getDevSelfCoefImag() const { return d_self_coef_i; }

    void calculate() override;

    void updateParticleCoordinates(const std::vector<double>& x_part,
                                   const std::vector<double>& y_part,
                                   const std::vector<double>& z_part) override;

    void updateFieldCoordinates(const std::vector<double>& x_field,
                                 const std::vector<double>& y_field,
                                 const std::vector<double>& z_field);

    void updateDipoles(const std::vector<double>& dip_x,
                       const std::vector<double>& dip_y,
                       const std::vector<double>& dip_z);

    void updateDipolesComplex(const std::vector<double>& dip_xr, const std::vector<double>& dip_xi,
                              const std::vector<double>& dip_yr, const std::vector<double>& dip_yi,
                              const std::vector<double>& dip_zr, const std::vector<double>& dip_zi);

    void updateQuadrupoles(const std::vector<double>& quad_1,
                           const std::vector<double>& quad_2,
                           const std::vector<double>& quad_3,
                           const std::vector<double>& quad_4,
                           const std::vector<double>& quad_5);

    void updateQuadrupolesComplex(const std::vector<double>& quad_1r, const std::vector<double>& quad_1i,
                                  const std::vector<double>& quad_2r, const std::vector<double>& quad_2i,
                                  const std::vector<double>& quad_3r, const std::vector<double>& quad_3i,
                                  const std::vector<double>& quad_4r, const std::vector<double>& quad_4i,
                                  const std::vector<double>& quad_5r, const std::vector<double>& quad_5i);

    // Helper to retrieve dipoles from GPU for validation
    void getDipolesHost(std::vector<double>& host_dip_x,
                        std::vector<double>& host_dip_y,
                        std::vector<double>& host_dip_z) const;

    void getDipolesComplexHost(std::vector<double>& host_dip_xr, std::vector<double>& host_dip_xi,
                               std::vector<double>& host_dip_yr, std::vector<double>& host_dip_yi,
                               std::vector<double>& host_dip_zr, std::vector<double>& host_dip_zi) const;

    void setSelfCoef(const std::vector<double>& self_coef_r,
                     const std::vector<double>& self_coef_i) override;
};

#endif // DIRECT_ELECTRIC_FIELD_H
