#ifndef MONODISPERSE_EWALD_ELECTRIC_FIELD_H
#define MONODISPERSE_EWALD_ELECTRIC_FIELD_H

#include "ewald_electric_field_base.h"
#include <vector>
#include <cstddef>
#include <memory>
#include <complex>

using Complex = std::complex<double>;

class Monodisperse_Ewald_Electric_Field : public Ewald_Electric_Field_Base {
private:
    // Members specific to monodisperse point solver
    double* d_self_perp = nullptr;
    double* d_khat = nullptr;
    double self_perp_val = 0.0;
    double self_G2_val = 0.0;

    void electricField() override;

public:
    Monodisperse_Ewald_Electric_Field(double box_x, double box_y, double box_z,
                                      double errortol,
                                      double xi,
                                      FieldCalcMode mode,
                                      double radius = 1.0,
                                      bool solve_quadrupoles = false,
                                      const std::vector<int>& quad_idxs = {});

    ~Monodisperse_Ewald_Electric_Field();

    Monodisperse_Ewald_Electric_Field(const Monodisperse_Ewald_Electric_Field&) = delete;
    Monodisperse_Ewald_Electric_Field& operator=(const Monodisperse_Ewald_Electric_Field&) = delete;

    double* getDevSelfPerp() const { return d_self_perp; }
    double* getDevKhat() const { return d_khat; }

    // Calculates real space tables (r_table, field_dip_1, field_dip_2)
    // and copies them to the GPU
    void computeRealSpaceTables();

    // Helper to copy real space tables from GPU to host vectors for validation
    void getRealSpaceTablesHost(std::vector<double>& host_r_table,
                                std::vector<double>& host_field_dip_1,
                                std::vector<double>& host_field_dip_2) const;

    // Performs precalculations from precalc.py and copies them to the GPU
    void computePrecalculations();

    // Performs wavevector-dependent Ewald scale precalculations on the GPU
    void computeScalePrecalcs();

    // Helper to retrieve precalculated vectors from GPU to host vectors for validation
    void getPrecalculationsHost(std::vector<int>& host_offset,
                                std::vector<double>& host_offsetxyz,
                                std::vector<double>& host_scale_coef,
                                std::vector<double>& host_khat) const;

    void updateParticleCoordinates(const std::vector<double>& x_part,
                                   const std::vector<double>& y_part,
                                   const std::vector<double>& z_part) override;

    void spreadPrecalcs() override;
    void contractPrecalcs() override;
    void realSpacePrecalcs() override;

    void getRealSpacePrecalcsHost(double& host_self_perp,
                                  std::vector<double>& host_perp,
                                  std::vector<double>& host_para) const;

    void spread(double* d_fE_grid) override;
    void scale(double* d_fE_grid) override;
    void contract(double* d_E_point, const double* d_Es_grid) override;
    void realSpace(double* d_E_point) override;

    void calculate() override;
};

#endif // MONODISPERSE_EWALD_ELECTRIC_FIELD_H
