#include "ewald_electric_field_base.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(val) { \
    if ((val) != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(val)) + " at line " + std::to_string(__LINE__)); \
    } \
}

Ewald_Electric_Field_Base::Ewald_Electric_Field_Base(double box_x, double box_y, double box_z,
                                                     double errortol, double xi, FieldCalcMode mode,
                                                     bool solve_quadrupoles, const std::vector<int>& quad_idxs,
                                                     PrecisionMode recip_precision)
    : Base_Electric_Field(mode, solve_quadrupoles, quad_idxs),
      box_x(box_x), box_y(box_y), box_z(box_z), errortol(errortol), xi(xi),
      recip_precision_setting(recip_precision),
      neighbor_list(std::make_unique<NeighborList>())
{
    use_recip_fp32 = determineRecipPrecisionMode(recip_precision);
    CUDA_CHECK(cudaStreamCreate(&stream_real));
    CUDA_CHECK(cudaStreamCreate(&stream_recip));
    CUDA_CHECK(cudaEventCreate(&event_recip_done));
}

bool Ewald_Electric_Field_Base::determineRecipPrecisionMode(PrecisionMode precision) {
    if (precision == PrecisionMode::MIXED || precision == PrecisionMode::FP32) return true;
    if (precision == PrecisionMode::DOUBLE || precision == PrecisionMode::FP64) return false;

    // AUTO mode defaults to mixed precision (FP32)
    return true;
}

Ewald_Electric_Field_Base::~Ewald_Electric_Field_Base() {
    if (d_r_table) { cudaFree(d_r_table); d_r_table = nullptr; }
    if (d_field_dip_1) { cudaFree(d_field_dip_1); d_field_dip_1 = nullptr; }
    if (d_field_dip_2) { cudaFree(d_field_dip_2); d_field_dip_2 = nullptr; }
    if (d_offset) { cudaFree(d_offset); d_offset = nullptr; }
    if (d_offsetxyz) { cudaFree(d_offsetxyz); d_offsetxyz = nullptr; }
    if (d_scale_coef) { cudaFree(d_scale_coef); d_scale_coef = nullptr; }
    if (d_spread_coef) { cudaFree(d_spread_coef); d_spread_coef = nullptr; }
    if (d_spread_idxs) { cudaFree(d_spread_idxs); d_spread_idxs = nullptr; }
    if (d_particle_index) { cudaFree(d_particle_index); d_particle_index = nullptr; }
    if (d_contract_coef) { cudaFree(d_contract_coef); d_contract_coef = nullptr; }
    if (d_contract_idxs) { cudaFree(d_contract_idxs); d_contract_idxs = nullptr; }
    if (d_perp) { cudaFree(d_perp); d_perp = nullptr; }
    if (d_para) { cudaFree(d_para); d_para = nullptr; }
    if (d_fE_grid) { cudaFree(d_fE_grid); d_fE_grid = nullptr; }
    if (fft_plan) { cufftDestroy((cufftHandle)fft_plan); fft_plan = 0; }
    if (d_field_quad_1) { cudaFree(d_field_quad_1); d_field_quad_1 = nullptr; }
    if (d_field_quad_2) { cudaFree(d_field_quad_2); d_field_quad_2 = nullptr; }
    if (d_field_quad_3) { cudaFree(d_field_quad_3); d_field_quad_3 = nullptr; }
    if (d_grad_quad_1) { cudaFree(d_grad_quad_1); d_grad_quad_1 = nullptr; }
    if (d_grad_quad_2) { cudaFree(d_grad_quad_2); d_grad_quad_2 = nullptr; }
    if (d_grad_quad_3) { cudaFree(d_grad_quad_3); d_grad_quad_3 = nullptr; }
    if (d_grad_quad_4) { cudaFree(d_grad_quad_4); d_grad_quad_4 = nullptr; }
    if (d_perp_Q) { cudaFree(d_perp_Q); d_perp_Q = nullptr; }
    if (d_para_Q) { cudaFree(d_para_Q); d_para_Q = nullptr; }
    if (d_Q3) { cudaFree(d_Q3); d_Q3 = nullptr; }
    if (d_G1) { cudaFree(d_G1); d_G1 = nullptr; }
    if (d_G2) { cudaFree(d_G2); d_G2 = nullptr; }
    if (d_G3) { cudaFree(d_G3); d_G3 = nullptr; }
    if (d_G4) { cudaFree(d_G4); d_G4 = nullptr; }
    if (d_fG_grid) { cudaFree(d_fG_grid); d_fG_grid = nullptr; }
    if (fft_plan_G) { cufftDestroy((cufftHandle)fft_plan_G); fft_plan_G = 0; }
    if (d_scale_coef_Q_imag) { cudaFree(d_scale_coef_Q_imag); d_scale_coef_Q_imag = nullptr; }
    if (d_scale_coef_GP_imag) { cudaFree(d_scale_coef_GP_imag); d_scale_coef_GP_imag = nullptr; }
    if (d_scale_coef_GQ_real) { cudaFree(d_scale_coef_GQ_real); d_scale_coef_GQ_real = nullptr; }
    if (d_Qfactor) { cudaFree(d_Qfactor); d_Qfactor = nullptr; }
    if (d_Qfactor_dot) { cudaFree(d_Qfactor_dot); d_Qfactor_dot = nullptr; }
    if (d_G_point) { cudaFree(d_G_point); d_G_point = nullptr; }
    if (stream_real) { cudaStreamDestroy(stream_real); stream_real = nullptr; }
    if (stream_recip) { cudaStreamDestroy(stream_recip); stream_recip = nullptr; }
    if (event_recip_done) { cudaEventDestroy(event_recip_done); event_recip_done = nullptr; }
}

void Ewald_Electric_Field_Base::computeNeighborList(int max_neighbors_per_particle) {
    neighbor_list->build(
        d_x_part, d_y_part, d_z_part,
        d_x_field, d_y_field, d_z_field,
        num_particles, num_field_points,
        box_x, box_y, box_z,
        rc, (mode == FieldCalcMode::SOLVER_AX),
        max_neighbors_per_particle
    );
}

// Verification methods
void Ewald_Electric_Field_Base::getNeighborListHost(std::vector<int>& host_list, std::vector<int>& host_counts) const {
    if (!neighbor_list) return;
    size_t list_size = num_particles * neighbor_list->get_max_neighbors();
    host_list.resize(list_size);
    host_counts.resize(num_particles);
    CUDA_CHECK(cudaMemcpy(host_list.data(), neighbor_list->get_list(), list_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_counts.data(), neighbor_list->get_counts(), num_particles * sizeof(int), cudaMemcpyDeviceToHost));
}

void Ewald_Electric_Field_Base::getRealSpaceTablesHost(std::vector<double>& host_r_table,
                                                      std::vector<double>& host_field_dip_1,
                                                      std::vector<double>& host_field_dip_2) const {
    host_r_table.resize(table_size);
    host_field_dip_1.resize(table_size);
    host_field_dip_2.resize(table_size);
    CUDA_CHECK(cudaMemcpy(host_r_table.data(), d_r_table, table_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_field_dip_1.data(), d_field_dip_1, table_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_field_dip_2.data(), d_field_dip_2, table_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void Ewald_Electric_Field_Base::getSpreadPrecalcsHost(std::vector<double>& host_spread_coef,
                                                     std::vector<int>& host_spread_idxs) const {
    host_spread_coef.resize(num_spread);
    host_spread_idxs.resize(num_spread);
    if (use_recip_fp32) {
        std::vector<float> temp(num_spread);
        CUDA_CHECK(cudaMemcpy(temp.data(), d_spread_coef, num_spread * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < num_spread; ++i) host_spread_coef[i] = static_cast<double>(temp[i]);
    } else {
        CUDA_CHECK(cudaMemcpy(host_spread_coef.data(), d_spread_coef, num_spread * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(host_spread_idxs.data(), d_spread_idxs, num_spread * sizeof(int), cudaMemcpyDeviceToHost));
}

void Ewald_Electric_Field_Base::getContractPrecalcsHost(std::vector<double>& host_E_point,
                                                       std::vector<int>& host_particle_index,
                                                       std::vector<double>& host_contract_coef,
                                                       std::vector<int>& host_contract_idxs) const {
    size_t num_targets = num_field_points > 0 ? num_field_points : num_particles;
    host_E_point.resize((num_targets * 3 + num_quads * 5) * 2);
    host_particle_index.resize(num_contract);
    host_contract_coef.resize(num_contract);
    host_contract_idxs.resize(num_contract);
    CUDA_CHECK(cudaMemcpy(host_E_point.data(), d_E_point, (num_targets * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_particle_index.data(), d_particle_index, num_contract * sizeof(int), cudaMemcpyDeviceToHost));
    if (use_recip_fp32) {
        std::vector<float> temp(num_contract);
        CUDA_CHECK(cudaMemcpy(temp.data(), d_contract_coef, num_contract * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < num_contract; ++i) host_contract_coef[i] = static_cast<double>(temp[i]);
    } else {
        CUDA_CHECK(cudaMemcpy(host_contract_coef.data(), d_contract_coef, num_contract * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(host_contract_idxs.data(), d_contract_idxs, num_contract * sizeof(int), cudaMemcpyDeviceToHost));
}

void Ewald_Electric_Field_Base::getRealSpacePrecalcsHost(double& host_self_perp,
                                                        std::vector<double>& host_perp,
                                                        std::vector<double>& host_para) const {
    size_t num_pairs = neighbor_list ? neighbor_list->get_num_pairs() : 0;
    host_perp.resize(num_pairs);
    host_para.resize(num_pairs);
    CUDA_CHECK(cudaMemcpy(&host_self_perp, d_perp, sizeof(double), cudaMemcpyDeviceToHost)); // Note: temporary default fallback
    CUDA_CHECK(cudaMemcpy(host_perp.data(), d_perp, num_pairs * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_para.data(), d_para, num_pairs * sizeof(double), cudaMemcpyDeviceToHost));
}
