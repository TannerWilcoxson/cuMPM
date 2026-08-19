#ifndef POLYDISPERSE_EWALD_ELECTRIC_FIELD_H
#define POLYDISPERSE_EWALD_ELECTRIC_FIELD_H

#include "ewald_electric_field_base.h"
#include <vector>
#include <cstddef>
#include <memory>
#include <complex>

using Complex = std::complex<double>;

class Polydisperse_Ewald_Electric_Field : public Ewald_Electric_Field_Base {
private:
    // Radii properties specific to polydisperse Ewald
    std::vector<double> h_radii;
    double* d_radii = nullptr;
    std::vector<double> unique_radii;
    size_t num_unique_radii = 0;
    size_t num_pairs_unique = 0;
    int* d_radius_idx = nullptr;
    int* d_col_ind = nullptr;
    double* d_self_perp_uniq = nullptr;
    double* d_self_G2_uniq = nullptr;

    double eta_scalar = 0.0;
    int P_support = 0;

    void* d_spread_coef_Q = nullptr;
    void* d_contract_coef_Q = nullptr;

    void electricField() override;

public:
    Polydisperse_Ewald_Electric_Field(double box_x, double box_y, double box_z,
                                      double errortol,
                                      double xi,
                                      FieldCalcMode mode,
                                      const std::vector<double>& particle_radii,
                                      bool solve_quadrupoles = false,
                                      const std::vector<int>& quad_idxs = {},
                                      PrecisionMode recip_precision = PrecisionMode::AUTO);

    ~Polydisperse_Ewald_Electric_Field();

    Polydisperse_Ewald_Electric_Field(const Polydisperse_Ewald_Electric_Field&) = delete;
    Polydisperse_Ewald_Electric_Field& operator=(const Polydisperse_Ewald_Electric_Field&) = delete;

    double getEta() const { return eta_scalar; }
    int getP() const { return P_support; }
    double* getDevGPoint() const { return d_G_point; }
    int* getDevQuadIdxs() const { return d_quad_idxs; }
    int* getDevQuadMap() const { return d_quad_map; }
    void* getDevSpreadCoefQ() const { return d_spread_coef_Q; }
    void* getDevContractCoefQ() const { return d_contract_coef_Q; }

    void computeRealSpaceTables();
    void computePrecalculations();
    void computeScalePrecalcs();
    void getPrecalculationsHost(std::vector<int>& host_offset,
                                std::vector<double>& host_offsetxyz,
                                std::vector<double>& host_scale_coef) const;

    void updateParticleCoordinates(const std::vector<double>& x_part,
                                   const std::vector<double>& y_part,
                                   const std::vector<double>& z_part) override;

    void spreadPrecalcs() override;
    void contractPrecalcs() override;
    void realSpacePrecalcs() override;

    void spread(void* d_fE_grid) override;
    void scale(void* d_fE_grid) override;
    void contract(double* d_E_point, const void* d_Es_grid) override;
    void realSpace(double* d_E_point) override;

    void calculate() override;
};

#endif // POLYDISPERSE_EWALD_ELECTRIC_FIELD_H
