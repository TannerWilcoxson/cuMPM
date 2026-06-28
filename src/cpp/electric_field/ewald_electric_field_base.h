#ifndef EWALD_ELECTRIC_FIELD_BASE_H
#define EWALD_ELECTRIC_FIELD_BASE_H

#include "electric_field.h"
#include <vector>
#include <cstddef>
#include <memory>
#include <complex>
#include "neighbor_list.h"

using Complex = std::complex<double>;

class Ewald_Electric_Field_Base : public Electric_Field {
protected:
    double* d_x_part = nullptr;
    double* d_y_part = nullptr;
    double* d_z_part = nullptr;
    size_t num_particles = 0;

    double* d_x_field = nullptr;
    double* d_y_field = nullptr;
    double* d_z_field = nullptr;
    size_t num_field_points = 0;

    double box_x = 0.0;
    double box_y = 0.0;
    double box_z = 0.0;
    double errortol = 0.0;
    double rc = 0.0;
    double xi = 0.5;
    bool calc_inter_dipole = true;

    int num_grid[3] = {0, 0, 0};
    double grid_spacing[3] = {0.0, 0.0, 0.0};
    double spectral_split[3] = {0.0, 0.0, 0.0};

    std::unique_ptr<NeighborList> neighbor_list;

    double* d_r_table = nullptr;
    double* d_field_dip_1 = nullptr;
    double* d_field_dip_2 = nullptr;
    size_t table_size = 0;

    int* d_offset = nullptr;
    double* d_offsetxyz = nullptr;
    double* d_scale_coef = nullptr;
    size_t num_offsets = 0;

    bool particles_updated = false;
    bool field_points_updated = false;

    double* d_dipoles = nullptr;
    bool dipoles_updated = false;

    double* d_self_coef_r = nullptr;
    double* d_self_coef_i = nullptr;

    double* d_spread_coef = nullptr;
    int* d_spread_idxs = nullptr;
    size_t num_spread = 0;

    double* d_E_point = nullptr;
    int* d_particle_index = nullptr;
    double* d_contract_coef = nullptr;
    int* d_contract_idxs = nullptr;
    size_t num_contract = 0;

    double* d_perp = nullptr;
    double* d_para = nullptr;
    double self_coef = 0.0;

    double* d_fE_grid = nullptr;
    double* d_fEs_grid = nullptr;
    int fft_plan = 0;

    bool solve_quadrupoles = false;
    std::vector<int> quad_idxs;
    int* d_quad_idxs = nullptr;
    int* d_quad_map = nullptr;
    size_t num_quads = 0;

    double* d_field_quad_1 = nullptr;
    double* d_field_quad_2 = nullptr;
    double* d_field_quad_3 = nullptr;
    double* d_grad_quad_1 = nullptr;
    double* d_grad_quad_2 = nullptr;
    double* d_grad_quad_3 = nullptr;
    double* d_grad_quad_4 = nullptr;

    double* d_perp_Q = nullptr;
    double* d_para_Q = nullptr;
    double* d_Q3 = nullptr;
    double* d_G1 = nullptr;
    double* d_G2 = nullptr;
    double* d_G3 = nullptr;
    double* d_G4 = nullptr;

    double* d_fG_grid = nullptr;
    double* d_fGs_grid = nullptr;
    int fft_plan_G = 0;

    double* d_scale_coef_Q_imag = nullptr;
    double* d_scale_coef_GP_imag = nullptr;
    double* d_scale_coef_GQ_real = nullptr;
    double* d_Qfactor = nullptr;
    double* d_Qfactor_dot = nullptr;

    double* d_G_point = nullptr;

    virtual void electricField() = 0;

public:
    Ewald_Electric_Field_Base(double box_x, double box_y, double box_z,
                              double errortol,
                              double xi,
                              bool calc_inter_dipole,
                              bool solve_quadrupoles = false,
                              const std::vector<int>& quad_idxs = {});

    virtual ~Ewald_Electric_Field_Base();

    Ewald_Electric_Field_Base(const Ewald_Electric_Field_Base&) = delete;
    Ewald_Electric_Field_Base& operator=(const Ewald_Electric_Field_Base&) = delete;

    // Getters / Setters
    double* getDevXPart() const { return d_x_part; }
    double* getDevYPart() const { return d_y_part; }
    double* getDevZPart() const { return d_z_part; }
    size_t getNumParticles() const { return num_particles; }

    double* getDevXField() const { return d_x_field; }
    double* getDevYField() const { return d_y_field; }
    double* getDevZField() const { return d_z_field; }
    size_t getNumFieldPoints() const { return num_field_points; }

    double getBoxX() const { return box_x; }
    double getBoxY() const { return box_y; }
    double getBoxZ() const { return box_z; }
    double getErrortol() const { return errortol; }
    double getRc() const { return rc; }
    double getXi() const { return xi; }
    bool getCalcInterDipole() const { return calc_inter_dipole; }
    double getSelfCoef() const { return self_coef; }

    bool getParticlesUpdated() const { return particles_updated; }
    bool getFieldPointsUpdated() const { return field_points_updated; }
    void clearParticlesUpdated() { particles_updated = false; }
    void clearFieldPointsUpdated() { field_points_updated = false; }

    double* getDevDipoles() const override { return d_dipoles; }
    double* getDevSelfCoefReal() const { return d_self_coef_r; }
    double* getDevSelfCoefImag() const { return d_self_coef_i; }
    bool getDipolesUpdated() const { return dipoles_updated; }
    void clearDipolesUpdated() { dipoles_updated = false; }

    double* getDevSpreadCoef() const { return d_spread_coef; }
    int* getDevSpreadIdxs() const { return d_spread_idxs; }
    size_t getNumSpread() const { return num_spread; }

    double* getDevEPoint() const override { return d_E_point; }
    int* getDevParticleIndex() const { return d_particle_index; }
    double* getDevContractCoef() const { return d_contract_coef; }
    int* getDevContractIdxs() const { return d_contract_idxs; }
    size_t getNumContract() const { return num_contract; }

    double* getDevPerp() const { return d_perp; }
    double* getDevPara() const { return d_para; }
    size_t getNumPairs() const { return neighbor_list ? neighbor_list->get_num_pairs() : 0; }

    const int* getNumGrid() const { return num_grid; }
    const double* getGridSpacing() const { return grid_spacing; }
    const double* getSpectralSplit() const { return spectral_split; }
    double* getDevFEGrid() const { return d_fE_grid; }
    double* getDevFEsGrid() const { return d_fEs_grid; }

    int* getDevNeighborList() const { return neighbor_list ? neighbor_list->get_list() : nullptr; }
    int* getDevNeighborCounts() const { return neighbor_list ? neighbor_list->get_counts() : nullptr; }
    int getMaxNeighbors() const { return neighbor_list ? neighbor_list->get_max_neighbors() : 0; }

    double* getDevRTable() const { return d_r_table; }
    double* getDevFieldDip1() const { return d_field_dip_1; }
    double* getDevFieldDip2() const { return d_field_dip_2; }
    size_t getTableSize() const { return table_size; }

    int* getDevOffset() const { return d_offset; }
    double* getDevOffsetxyz() const { return d_offsetxyz; }
    double* getDevScaleCoef() const { return d_scale_coef; }
    size_t getNumOffsets() const { return num_offsets; }

    bool getSolveQuadrupoles() const { return solve_quadrupoles; }
    size_t getNumQuads() const { return num_quads; }

    // Verification methods
    void getNeighborListHost(std::vector<int>& host_list, std::vector<int>& host_counts) const;
    void getRealSpaceTablesHost(std::vector<double>& host_r_table,
                                std::vector<double>& host_field_dip_1,
                                std::vector<double>& host_field_dip_2) const;
    void getDipolesHost(std::vector<double>& host_dip_x,
                        std::vector<double>& host_dip_y,
                        std::vector<double>& host_dip_z) const;
    void getDipolesComplexHost(std::vector<double>& host_dip_xr, std::vector<double>& host_dip_xi,
                               std::vector<double>& host_dip_yr, std::vector<double>& host_dip_yi,
                               std::vector<double>& host_dip_zr, std::vector<double>& host_dip_zi) const;
    void getSpreadPrecalcsHost(std::vector<double>& host_spread_coef,
                               std::vector<int>& host_spread_idxs) const;
    void getContractPrecalcsHost(std::vector<double>& host_E_point,
                                 std::vector<int>& host_particle_index,
                                 std::vector<double>& host_contract_coef,
                                 std::vector<int>& host_contract_idxs) const;
    void getRealSpacePrecalcsHost(double& host_self_perp,
                                  std::vector<double>& host_perp,
                                  std::vector<double>& host_para) const;
    std::vector<Complex> getEPointHost() const;
    void clearEPoint();

    // Coordinate update methods
    void computeNeighborList(int max_neighbors_per_particle = 128);

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

    void setSelfCoef(const std::vector<double>& self_coef_r, const std::vector<double>& self_coef_i) override;
    void setSelfCoef(double val_r, double val_i = 0.0);

    virtual void spreadPrecalcs() = 0;
    virtual void contractPrecalcs() = 0;
    virtual void realSpacePrecalcs() = 0;

    virtual void spread(double* d_fE_grid) = 0;
    virtual void scale(double* d_fE_grid) = 0;
    virtual void contract(double* d_E_point, const double* d_Es_grid) = 0;
    virtual void realSpace(double* d_E_point) = 0;

    void calculate() override = 0;
};

#endif // EWALD_ELECTRIC_FIELD_BASE_H
