#ifndef EWALD_ELECTRIC_FIELD_H
#define EWALD_ELECTRIC_FIELD_H

#include "electric_field.h"
#include <vector>
#include <cstddef>
#include <memory>
#include "neighbor_list.h"

class Ewald_Electric_Field : public Electric_Field {
private:
    double* d_x_part = nullptr; // GPU device pointer for particle X-coordinates
    double* d_y_part = nullptr; // GPU device pointer for particle Y-coordinates
    double* d_z_part = nullptr; // GPU device pointer for particle Z-coordinates
    size_t num_particles = 0;

    double* d_x_field = nullptr; // GPU device pointer for field point X-coordinates
    double* d_y_field = nullptr; // GPU device pointer for field point Y-coordinates
    double* d_z_field = nullptr; // GPU device pointer for field point Z-coordinates
    size_t num_field_points = 0;

    // Box dimensions and Ewald parameters
    double box_x = 0.0;
    double box_y = 0.0;
    double box_z = 0.0;
    double errortol = 0.0;
    double rc = 0.0; // Real space cutoff computed from errortol and xi
    double xi = 0.5;
    bool calc_inter_dipole = true;

    // Grid properties
    int num_grid[3] = {0, 0, 0};
    double grid_spacing[3] = {0.0, 0.0, 0.0};
    double spectral_split[3] = {0.0, 0.0, 0.0};

    // Neighbor list manager
    std::unique_ptr<NeighborList> neighbor_list;

    // Device pointers for real space tables
    double* d_r_table = nullptr;
    double* d_field_dip_1 = nullptr;
    double* d_field_dip_2 = nullptr;
    size_t table_size = 0;

    // Device pointers for Ewald precalculations
    int* d_offset = nullptr;
    double* d_offsetxyz = nullptr;
    double* d_scale_coef = nullptr;
    double* d_khat = nullptr;
    size_t num_offsets = 0;

    // Coordinate update tracking flags
    bool particles_updated = false;
    bool field_points_updated = false;

    // GPU device pointer for contiguous complex dipoles
    // Layout: num_particles * 3 components (x, y, z) * 2 (real, imag)
    double* d_dipoles = nullptr;
    bool dipoles_updated = false;

    // GPU device pointers for complex self coefficients
    double* d_self_coef_r = nullptr;
    double* d_self_coef_i = nullptr;

    // GPU device pointers for spread precalcs
    double* d_spread_coef = nullptr;
    int* d_spread_idxs = nullptr;
    size_t num_spread = 0;

    // GPU device pointers for contract precalcs
    double* d_E_point = nullptr;
    int* d_particle_index = nullptr;
    double* d_contract_coef = nullptr;
    int* d_contract_idxs = nullptr;
    size_t num_contract = 0;

    // GPU device pointers for real space precalcs
    double* d_self_perp = nullptr;
    double* d_perp = nullptr;
    double* d_para = nullptr;
    double self_coef = 0.0;

    // Grid and FFT members
    double* d_fE_grid = nullptr;
    double* d_fEs_grid = nullptr;
    int fft_plan = 0;

    // Private method for the Ewald calculation pipeline
    void electricField();

public:
    // Constructor: Allocates Ewald precalculations, tables, and FFT plans
    Ewald_Electric_Field(double box_x, double box_y, double box_z,
                         double errortol,
                         double xi,
                         bool calc_inter_dipole);

    // Destructor: Frees GPU memory
    ~Ewald_Electric_Field();

    // Disable copy constructor and copy assignment operator to prevent double-free
    Ewald_Electric_Field(const Ewald_Electric_Field&) = delete;
    Ewald_Electric_Field& operator=(const Ewald_Electric_Field&) = delete;

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

    double* getDevSelfPerp() const { return d_self_perp; }
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
    double* getDevKhat() const { return d_khat; }
    size_t getNumOffsets() const { return num_offsets; }

    // Neighbor list calculation on GPU
    void computeNeighborList(int max_neighbors_per_particle = 128);

    // Helper to retrieve neighbor list on host for verification/use
    void getNeighborListHost(std::vector<int>& host_list, std::vector<int>& host_counts) const;

    // Calculates real space tables (r_table, field_dip_1, field_dip_2)
    // and copies them to the GPU
    void computeRealSpaceTables();

    // Helper to copy real space tables from GPU to host vectors for validation
    void getRealSpaceTablesHost(std::vector<double>& host_r_table,
                                std::vector<double>& host_field_dip_1,
                                std::vector<double>& host_field_dip_2) const;

    // Performs precalculations from precalc.py and copies them to the GPU
    void computePrecalculations();

    // Helper to retrieve precalculated vectors from GPU to host vectors for validation
    void getPrecalculationsHost(std::vector<int>& host_offset,
                                std::vector<double>& host_offsetxyz,
                                std::vector<double>& host_scale_coef,
                                std::vector<double>& host_khat) const;

    // Update coordinate methods
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

    // Helper to retrieve dipoles from GPU for validation
    void getDipolesHost(std::vector<double>& host_dip_x,
                        std::vector<double>& host_dip_y,
                        std::vector<double>& host_dip_z) const;

    void getDipolesComplexHost(std::vector<double>& host_dip_xr, std::vector<double>& host_dip_xi,
                               std::vector<double>& host_dip_yr, std::vector<double>& host_dip_yi,
                               std::vector<double>& host_dip_zr, std::vector<double>& host_dip_zi) const;

    void setSelfCoef(const std::vector<double>& self_coef_r, const std::vector<double>& self_coef_i) override;
    void setSelfCoef(double val_r, double val_i = 0.0);

    // New precalculation methods
    void spreadPrecalcs();
    void contractPrecalcs();
    void realSpacePrecalcs();

    // Helpers to retrieve precalculations from GPU for validation
    void getSpreadPrecalcsHost(std::vector<double>& host_spread_coef,
                               std::vector<int>& host_spread_idxs) const;

    void getContractPrecalcsHost(std::vector<double>& host_E_point,
                                 std::vector<int>& host_particle_index,
                                 std::vector<double>& host_contract_coef,
                                 std::vector<int>& host_contract_idxs) const;

    void getRealSpacePrecalcsHost(double& host_self_perp,
                                  std::vector<double>& host_perp,
                                  std::vector<double>& host_para) const;

    // GPU calculations from main_calcs.py
    void spread(double* d_fE_grid);
    void scale(double* d_fE_grid);
    void contract(double* d_E_point, const double* d_Es_grid);
    void realSpace(double* d_E_point);

    // Public method to run Ewald calculation pipeline
    void calculate() override;
};

#endif // EWALD_ELECTRIC_FIELD_H
