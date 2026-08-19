#ifndef EWALD_ELECTRIC_FIELD_BASE_H
#define EWALD_ELECTRIC_FIELD_BASE_H

#include "electric_field.h"
#include "direct_electric_field.h"
#include <vector>
#include <cstddef>
#include <memory>
#include <complex>
#include <cuda_runtime.h>
#include "neighbor_list.h"

using Complex = std::complex<double>;

class Ewald_Electric_Field_Base : public Base_Electric_Field {
protected:
    double box_x = 0.0;
    double box_y = 0.0;
    double box_z = 0.0;
    double errortol = 0.0;
    double rc = 0.0;
    double xi = 0.5;

    PrecisionMode recip_precision_setting = PrecisionMode::AUTO;
    bool use_recip_fp32 = false;

    bool determineRecipPrecisionMode(PrecisionMode mode);

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
    void* d_scale_coef = nullptr;
    size_t num_offsets = 0;

    void* d_spread_coef = nullptr;
    int* d_spread_idxs = nullptr;
    size_t num_spread = 0;

    int* d_particle_index = nullptr;
    void* d_contract_coef = nullptr;
    int* d_contract_idxs = nullptr;
    size_t num_contract = 0;

    double* d_perp = nullptr;
    double* d_para = nullptr;
    double self_coef = 0.0;

    void* d_fE_grid = nullptr;
    int fft_plan = 0;

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

    void* d_fG_grid = nullptr;
    int fft_plan_G = 0;

    void* d_scale_coef_Q_imag = nullptr;
    void* d_scale_coef_GP_imag = nullptr;
    void* d_scale_coef_GQ_real = nullptr;
    void* d_Qfactor = nullptr;
    void* d_Qfactor_dot = nullptr;

    double* d_G_point = nullptr;

    cudaStream_t stream_real = nullptr;
    cudaStream_t stream_recip = nullptr;
    cudaEvent_t event_recip_done = nullptr;
    bool use_async_streams = false;

    virtual void electricField() = 0;

public:
    Ewald_Electric_Field_Base(double box_x, double box_y, double box_z,
                              double errortol,
                              double xi,
                              FieldCalcMode mode,
                              bool solve_quadrupoles = false,
                              const std::vector<int>& quad_idxs = {},
                              PrecisionMode recip_precision = PrecisionMode::AUTO);

    virtual ~Ewald_Electric_Field_Base();

    Ewald_Electric_Field_Base(const Ewald_Electric_Field_Base&) = delete;
    Ewald_Electric_Field_Base& operator=(const Ewald_Electric_Field_Base&) = delete;

    // Getters / Setters
    double getBoxX() const { return box_x; }
    double getBoxY() const { return box_y; }
    double getBoxZ() const { return box_z; }
    double getErrortol() const { return errortol; }
    double getRc() const { return rc; }
    double getXi() const { return xi; }
    double getSelfCoef() const { return self_coef; }

    bool isUsingRecipFP32() const { return use_recip_fp32; }
    PrecisionMode getRecipPrecisionMode() const { return recip_precision_setting; }

    cudaStream_t getStreamReal() const { return stream_real; }
    cudaStream_t getStreamRecip() const { return stream_recip; }
    cudaEvent_t getEventRecipDone() const { return event_recip_done; }
    bool isUsingAsyncStreams() const { return use_async_streams; }
    void setUseAsyncStreams(bool enable) { use_async_streams = enable; }

    double k_x = 0.0;
    double k_y = 0.0;
    bool kt_updated = false;

    void setBlochWavevector(double kx, double ky) {
        if (k_x != kx || k_y != ky) {
            k_x = kx;
            k_y = ky;
            kt_updated = true;
        }
    }

    void* getDevSpreadCoef() const { return d_spread_coef; }
    int* getDevSpreadIdxs() const { return d_spread_idxs; }
    size_t getNumSpread() const { return num_spread; }

    int* getDevParticleIndex() const { return d_particle_index; }
    void* getDevContractCoef() const { return d_contract_coef; }
    int* getDevContractIdxs() const { return d_contract_idxs; }
    size_t getNumContract() const { return num_contract; }

    double* getDevPerp() const { return d_perp; }
    double* getDevPara() const { return d_para; }
    size_t getNumPairs() const { return neighbor_list ? neighbor_list->get_num_pairs() : 0; }

    const int* getNumGrid() const { return num_grid; }
    const double* getGridSpacing() const { return grid_spacing; }
    const double* getSpectralSplit() const { return spectral_split; }
    void* getDevFEGrid() const { return d_fE_grid; }

    int* getDevNeighborList() const { return neighbor_list ? neighbor_list->get_list() : nullptr; }
    int* getDevNeighborCounts() const { return neighbor_list ? neighbor_list->get_counts() : nullptr; }
    int getMaxNeighbors() const { return neighbor_list ? neighbor_list->get_max_neighbors() : 0; }

    double* getDevRTable() const { return d_r_table; }
    double* getDevFieldDip1() const { return d_field_dip_1; }
    double* getDevFieldDip2() const { return d_field_dip_2; }
    size_t getTableSize() const { return table_size; }

    int* getDevOffset() const { return d_offset; }
    double* getDevOffsetxyz() const { return d_offsetxyz; }
    void* getDevScaleCoef() const { return d_scale_coef; }
    size_t getNumOffsets() const { return num_offsets; }

    // Verification methods
    void getNeighborListHost(std::vector<int>& host_list, std::vector<int>& host_counts) const;
    void getRealSpaceTablesHost(std::vector<double>& host_r_table,
                                std::vector<double>& host_field_dip_1,
                                std::vector<double>& host_field_dip_2) const;
    void getSpreadPrecalcsHost(std::vector<double>& host_spread_coef,
                               std::vector<int>& host_spread_idxs) const;
    void getContractPrecalcsHost(std::vector<double>& host_E_point,
                                 std::vector<int>& host_particle_index,
                                 std::vector<double>& host_contract_coef,
                                 std::vector<int>& host_contract_idxs) const;
    void getRealSpacePrecalcsHost(double& host_self_perp,
                                  std::vector<double>& host_perp,
                                  std::vector<double>& host_para) const;

    // Coordinate update methods
    void computeNeighborList(int max_neighbors_per_particle = 128);

    virtual void spreadPrecalcs() = 0;
    virtual void contractPrecalcs() = 0;
    virtual void realSpacePrecalcs() = 0;

    virtual void spread(void* d_fE_grid) = 0;
    virtual void scale(void* d_fE_grid) = 0;
    virtual void contract(double* d_E_point, const void* d_Es_grid) = 0;
    virtual void realSpace(double* d_E_point) = 0;

    void calculate() override = 0;
};

#endif // EWALD_ELECTRIC_FIELD_BASE_H
