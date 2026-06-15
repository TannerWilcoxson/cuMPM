#ifndef POLYDISPERSE_ELECTRIC_FIELD_H
#define POLYDISPERSE_ELECTRIC_FIELD_H

#include "electric_field.h"
#include <vector>
#include <cstddef>
#include <string>

class Polydisperse_Electric_Field : public Electric_Field {
private:
    double* d_x_part = nullptr;
    double* d_y_part = nullptr;
    double* d_z_part = nullptr;
    size_t num_particles = 0;

    double* d_x_field = nullptr;
    double* d_y_field = nullptr;
    double* d_z_field = nullptr;
    size_t num_field_points = 0;

    // Box dimensions and Ewald parameters
    double box_x = 0.0;
    double box_y = 0.0;
    double box_z = 0.0;
    double errortol = 0.0;
    double rc = 0.0;
    double xi = 0.5;
    bool calc_inter_dipole = true;

    // Radii properties
    std::vector<double> h_radii;
    double* d_radii = nullptr; // GPU pointer for particle radii
    std::vector<double> unique_radii;
    size_t num_unique_radii = 0;
    size_t num_pairs_unique = 0; // M*(M+1)/2
    int* d_radius_idx = nullptr; // GPU pointer for each particle's radius index
    int* d_col_ind = nullptr;    // GPU pointer for M x M column index map
    double* d_self_perp_uniq = nullptr; // GPU pointer for self perp for each unique radius

    // Grid properties
    int num_grid[3] = {0, 0, 0};
    double grid_spacing[3] = {0.0, 0.0, 0.0};
    double eta_scalar = 0.0;
    int P_support = 0;

    // Neighbor list device pointers
    int* d_neighbor_list = nullptr;
    int* d_neighbor_counts = nullptr;
    int max_neighbors = 0;

    // Device pointers for real space tables
    double* d_r_table = nullptr;
    double* d_field_dip_1 = nullptr; // Ep_perp (table_size x num_pairs_unique)
    double* d_field_dip_2 = nullptr; // Ep_para (table_size x num_pairs_unique)
    size_t table_size = 0;

    // Device pointers for Ewald precalculations
    int* d_offset = nullptr;
    double* d_offsetxyz = nullptr;
    double* d_scale_coef = nullptr; // Scalar grid scale factors
    size_t num_offsets = 0;

    // Coordinate update tracking flags
    bool particles_updated = false;
    bool field_points_updated = false;

    // GPU device pointer for contiguous complex dipoles
    // Layout: num_particles * 3 components * 2 (real, imag)
    double* d_dipoles = nullptr;
    bool dipoles_updated = false;

    // GPU device pointers for complex self coefficients
    double* d_self_coef_r = nullptr;
    double* d_self_coef_i = nullptr;

    // GPU device pointers for spread precalcs
    double* d_spread_coef = nullptr; // Stores 3-vector coefficients (num_spread * 3)
    int* d_spread_idxs = nullptr;     // Stores grid indices (num_spread * 3)
    size_t num_spread = 0;

    // GPU device pointers for contract precalcs
    double* d_E_point = nullptr;
    int* d_particle_index = nullptr;
    double* d_contract_coef = nullptr; // Stores 3-vector coefficients (num_contract * 3)
    int* d_contract_idxs = nullptr;     // Stores grid indices (num_contract * 3)
    size_t num_contract = 0;

    // GPU device pointers for real space precalcs
    double* d_perp = nullptr;
    double* d_para = nullptr;
    int* d_particle_offsets = nullptr;
    size_t num_pairs = 0;

    // Grid and FFT members
    double* d_fE_grid = nullptr;  // Scalar grid of size grid_voxels (complex, Z2Z format)
    double* d_fEs_grid = nullptr; // Shuffled scalar grid
    int fft_plan = 0;

    void electricField();

public:
    Polydisperse_Electric_Field(double box_x, double box_y, double box_z,
                                double errortol,
                                double xi,
                                bool calc_inter_dipole,
                                const std::vector<double>& particle_radii);

    ~Polydisperse_Electric_Field();

    Polydisperse_Electric_Field(const Polydisperse_Electric_Field&) = delete;
    Polydisperse_Electric_Field& operator=(const Polydisperse_Electric_Field&) = delete;

    // Getters
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
    size_t getNumPairs() const { return num_pairs; }

    const int* getNumGrid() const { return num_grid; }
    const double* getGridSpacing() const { return grid_spacing; }
    double* getDevFEGrid() const { return d_fE_grid; }
    double* getDevFEsGrid() const { return d_fEs_grid; }
    double getEta() const { return eta_scalar; }
    int getP() const { return P_support; }

    int* getDevNeighborList() const { return d_neighbor_list; }
    int* getDevNeighborCounts() const { return d_neighbor_counts; }
    int getMaxNeighbors() const { return max_neighbors; }

    double* getDevRTable() const { return d_r_table; }
    double* getDevFieldDip1() const { return d_field_dip_1; }
    double* getDevFieldDip2() const { return d_field_dip_2; }
    size_t getTableSize() const { return table_size; }

    int* getDevOffset() const { return d_offset; }
    double* getDevOffsetxyz() const { return d_offsetxyz; }
    double* getDevScaleCoef() const { return d_scale_coef; }
    size_t getNumOffsets() const { return num_offsets; }

    void computeNeighborList(int max_neighbors_per_particle = 128);
    void computeRealSpaceTables();
    void computePrecalculations();

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

    void setSelfCoef(const std::vector<double>& self_coef_r, const std::vector<double>& self_coef_i) override;
    void setSelfCoef(double val_r, double val_i = 0.0);

    void spreadPrecalcs();
    void contractPrecalcs();
    void realSpacePrecalcs();

    void spread(double* d_fE_grid);
    void scale(double* d_fE_grid);
    void contract(double* d_E_point, const double* d_Es_grid);
    void realSpace(double* d_E_point);

    void calculate() override;
};

#endif // POLYDISPERSE_ELECTRIC_FIELD_H
