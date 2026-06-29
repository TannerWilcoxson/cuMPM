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
                                                     bool solve_quadrupoles, const std::vector<int>& quad_idxs)
    : box_x(box_x), box_y(box_y), box_z(box_z), errortol(errortol), xi(xi),
      mode(mode), solve_quadrupoles(solve_quadrupoles),
      quad_idxs(quad_idxs), neighbor_list(std::make_unique<NeighborList>())
{
}

Ewald_Electric_Field_Base::~Ewald_Electric_Field_Base() {
    if (d_x_part) cudaFree(d_x_part);
    if (d_y_part) cudaFree(d_y_part);
    if (d_z_part) cudaFree(d_z_part);
    if (d_x_field && d_x_field != d_x_part) cudaFree(d_x_field);
    if (d_y_field && d_y_field != d_y_part) cudaFree(d_y_field);
    if (d_z_field && d_z_field != d_z_part) cudaFree(d_z_field);
    if (d_r_table) cudaFree(d_r_table);
    if (d_field_dip_1) cudaFree(d_field_dip_1);
    if (d_field_dip_2) cudaFree(d_field_dip_2);
    if (d_offset) cudaFree(d_offset);
    if (d_offsetxyz) cudaFree(d_offsetxyz);
    if (d_scale_coef) cudaFree(d_scale_coef);
    if (d_dipoles) cudaFree(d_dipoles);
    if (d_self_coef_r) cudaFree(d_self_coef_r);
    if (d_self_coef_i) cudaFree(d_self_coef_i);
    if (d_spread_coef) cudaFree(d_spread_coef);
    if (d_spread_idxs) cudaFree(d_spread_idxs);
    if (d_E_point) cudaFree(d_E_point);
    if (d_particle_index) cudaFree(d_particle_index);
    if (d_contract_coef) cudaFree(d_contract_coef);
    if (d_contract_idxs) cudaFree(d_contract_idxs);
    if (d_perp) cudaFree(d_perp);
    if (d_para) cudaFree(d_para);
    if (d_fE_grid) cudaFree(d_fE_grid);
    if (d_fEs_grid) cudaFree(d_fEs_grid);
    if (fft_plan) cufftDestroy((cufftHandle)fft_plan);
    if (d_quad_idxs) cudaFree(d_quad_idxs);
    if (d_quad_map) cudaFree(d_quad_map);
    if (d_field_quad_1) cudaFree(d_field_quad_1);
    if (d_field_quad_2) cudaFree(d_field_quad_2);
    if (d_field_quad_3) cudaFree(d_field_quad_3);
    if (d_grad_quad_1) cudaFree(d_grad_quad_1);
    if (d_grad_quad_2) cudaFree(d_grad_quad_2);
    if (d_grad_quad_3) cudaFree(d_grad_quad_3);
    if (d_grad_quad_4) cudaFree(d_grad_quad_4);
    if (d_perp_Q) cudaFree(d_perp_Q);
    if (d_para_Q) cudaFree(d_para_Q);
    if (d_Q3) cudaFree(d_Q3);
    if (d_G1) cudaFree(d_G1);
    if (d_G2) cudaFree(d_G2);
    if (d_G3) cudaFree(d_G3);
    if (d_G4) cudaFree(d_G4);
    if (d_fG_grid) cudaFree(d_fG_grid);
    if (d_fGs_grid) cudaFree(d_fGs_grid);
    if (fft_plan_G) cufftDestroy((cufftHandle)fft_plan_G);
    if (d_scale_coef_Q_imag) cudaFree(d_scale_coef_Q_imag);
    if (d_scale_coef_GP_imag) cudaFree(d_scale_coef_GP_imag);
    if (d_scale_coef_GQ_real) cudaFree(d_scale_coef_GQ_real);
    if (d_Qfactor) cudaFree(d_Qfactor);
    if (d_Qfactor_dot) cudaFree(d_Qfactor_dot);
    if (d_G_point) cudaFree(d_G_point);
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

void Ewald_Electric_Field_Base::updateParticleCoordinates(const std::vector<double>& x_part,
                                                         const std::vector<double>& y_part,
                                                         const std::vector<double>& z_part) {
    if (x_part.size() != y_part.size() || x_part.size() != z_part.size()) {
        throw std::invalid_argument("Input coordinate vectors must have the exact same size.");
    }
    size_t new_num_particles = x_part.size();
    if (new_num_particles == 0) return;

    if (d_quad_idxs) { cudaFree(d_quad_idxs); d_quad_idxs = nullptr; }
    if (d_quad_map) { cudaFree(d_quad_map); d_quad_map = nullptr; }

    if (solve_quadrupoles) {
        num_quads = quad_idxs.empty() ? new_num_particles : quad_idxs.size();
        CUDA_CHECK(cudaMalloc(&d_quad_idxs, num_quads * sizeof(int)));
        std::vector<int> host_quad_idxs(num_quads);
        if (quad_idxs.empty()) {
            for (size_t i = 0; i < num_quads; ++i) {
                host_quad_idxs[i] = i;
            }
        } else {
            host_quad_idxs = quad_idxs;
        }
        CUDA_CHECK(cudaMemcpy(d_quad_idxs, host_quad_idxs.data(), num_quads * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_quad_map, new_num_particles * sizeof(int)));
        std::vector<int> host_quad_map(new_num_particles, -1);
        for (size_t q = 0; q < num_quads; ++q) {
            int p_idx = host_quad_idxs[q];
            if (p_idx >= 0 && p_idx < static_cast<int>(new_num_particles)) {
                host_quad_map[p_idx] = q;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_quad_map, host_quad_map.data(), new_num_particles * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        num_quads = 0;
    }

    if (new_num_particles != num_particles || d_dipoles == nullptr) {
        if (d_x_part) cudaFree(d_x_part);
        if (d_y_part) cudaFree(d_y_part);
        if (d_z_part) cudaFree(d_z_part);
        if (d_dipoles) cudaFree(d_dipoles);
        if (d_self_coef_r) { cudaFree(d_self_coef_r); d_self_coef_r = nullptr; }
        if (d_self_coef_i) { cudaFree(d_self_coef_i); d_self_coef_i = nullptr; }

        num_particles = new_num_particles;
        size_t size_part_bytes = num_particles * sizeof(double);
        size_t size_dipoles_bytes = (num_particles * 3 + num_quads * 5) * 2 * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_x_part, size_part_bytes));
        CUDA_CHECK(cudaMalloc(&d_y_part, size_part_bytes));
        CUDA_CHECK(cudaMalloc(&d_z_part, size_part_bytes));
        CUDA_CHECK(cudaMalloc(&d_dipoles, size_dipoles_bytes));
        CUDA_CHECK(cudaMemset(d_dipoles, 0, size_dipoles_bytes));

        CUDA_CHECK(cudaMalloc(&d_self_coef_r, size_part_bytes));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, size_part_bytes));

        std::vector<double> host_sc_r(num_particles, self_coef);
        std::vector<double> host_sc_i(num_particles, 0.0);
        CUDA_CHECK(cudaMemcpy(d_self_coef_r, host_sc_r.data(), size_part_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_self_coef_i, host_sc_i.data(), size_part_bytes, cudaMemcpyHostToDevice));
    }

    size_t size_part_bytes = num_particles * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_x_part, x_part.data(), size_part_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_part, y_part.data(), size_part_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_part, z_part.data(), size_part_bytes, cudaMemcpyHostToDevice));

    particles_updated = true;

    if (mode == FieldCalcMode::SOLVER_AX) {
        d_x_field = d_x_part;
        d_y_field = d_y_part;
        d_z_field = d_z_part;
        num_field_points = num_particles;
        field_points_updated = true;
    }
}

void Ewald_Electric_Field_Base::updateFieldCoordinates(const std::vector<double>& x_field,
                                                      const std::vector<double>& y_field,
                                                      const std::vector<double>& z_field) {
    if (mode == FieldCalcMode::SOLVER_AX) {
        throw std::runtime_error("Ewald_Electric_Field_Base: Cannot update field coordinates when SOLVER_AX mode is active.");
    }
    num_field_points = x_field.size();
    size_t size_bytes = num_field_points * sizeof(double);

    if (d_x_field) cudaFree(d_x_field);
    if (d_y_field) cudaFree(d_y_field);
    if (d_z_field) cudaFree(d_z_field);

    CUDA_CHECK(cudaMalloc(&d_x_field, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_y_field, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_z_field, size_bytes));

    CUDA_CHECK(cudaMemcpy(d_x_field, x_field.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_field, y_field.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_field, z_field.data(), size_bytes, cudaMemcpyHostToDevice));

    field_points_updated = true;
}

void Ewald_Electric_Field_Base::updateDipoles(const std::vector<double>& dip_x,
                                             const std::vector<double>& dip_y,
                                             const std::vector<double>& dip_z) {
    if (dip_x.size() != num_particles || dip_y.size() != num_particles || dip_z.size() != num_particles) {
        throw std::invalid_argument("Input dipole component vectors must have size matching num_particles.");
    }
    std::vector<double> host_dipoles((num_particles * 3 + num_quads * 5) * 2, 0.0);
    if (d_dipoles != nullptr) {
        CUDA_CHECK(cudaMemcpy(host_dipoles.data(), d_dipoles, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    }
    for (size_t i = 0; i < num_particles; ++i) {
        host_dipoles[(i * 3 + 0) * 2 + 0] = dip_x[i];
        host_dipoles[(i * 3 + 0) * 2 + 1] = 0.0;
        host_dipoles[(i * 3 + 1) * 2 + 0] = dip_y[i];
        host_dipoles[(i * 3 + 1) * 2 + 1] = 0.0;
        host_dipoles[(i * 3 + 2) * 2 + 0] = dip_z[i];
        host_dipoles[(i * 3 + 2) * 2 + 1] = 0.0;
    }
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dipoles.data(), (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyHostToDevice));
    dipoles_updated = true;
}

void Ewald_Electric_Field_Base::updateDipolesComplex(const std::vector<double>& dip_xr, const std::vector<double>& dip_xi,
                                                    const std::vector<double>& dip_yr, const std::vector<double>& dip_yi,
                                                    const std::vector<double>& dip_zr, const std::vector<double>& dip_zi) {
    if (dip_xr.size() != num_particles || dip_xi.size() != num_particles ||
        dip_yr.size() != num_particles || dip_yi.size() != num_particles ||
        dip_zr.size() != num_particles || dip_zi.size() != num_particles) {
        throw std::invalid_argument("Input dipole component vectors must have size matching num_particles.");
    }
    std::vector<double> host_dipoles((num_particles * 3 + num_quads * 5) * 2, 0.0);
    if (d_dipoles != nullptr) {
        CUDA_CHECK(cudaMemcpy(host_dipoles.data(), d_dipoles, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    }
    for (size_t i = 0; i < num_particles; ++i) {
        host_dipoles[(i * 3 + 0) * 2 + 0] = dip_xr[i];
        host_dipoles[(i * 3 + 0) * 2 + 1] = dip_xi[i];
        host_dipoles[(i * 3 + 1) * 2 + 0] = dip_yr[i];
        host_dipoles[(i * 3 + 1) * 2 + 1] = dip_yi[i];
        host_dipoles[(i * 3 + 2) * 2 + 0] = dip_zr[i];
        host_dipoles[(i * 3 + 2) * 2 + 1] = dip_zi[i];
    }
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dipoles.data(), (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyHostToDevice));
    dipoles_updated = true;
}

void Ewald_Electric_Field_Base::updateQuadrupoles(const std::vector<double>& quad_1,
                                                 const std::vector<double>& quad_2,
                                                 const std::vector<double>& quad_3,
                                                 const std::vector<double>& quad_4,
                                                 const std::vector<double>& quad_5) {
    if (!solve_quadrupoles || num_quads == 0) {
        throw std::runtime_error("updateQuadrupoles: Quadrupoles are not enabled for this solver.");
    }
    if (quad_1.size() != num_quads || quad_2.size() != num_quads || quad_3.size() != num_quads ||
        quad_4.size() != num_quads || quad_5.size() != num_quads) {
        throw std::invalid_argument("Input quadrupole component vectors must have size matching num_quads.");
    }
    std::vector<double> host_dipoles((num_particles * 3 + num_quads * 5) * 2, 0.0);
    if (d_dipoles != nullptr) {
        CUDA_CHECK(cudaMemcpy(host_dipoles.data(), d_dipoles, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    }
    size_t quad_start_offset = num_particles * 3 * 2;
    for (size_t q = 0; q < num_quads; ++q) {
        host_dipoles[quad_start_offset + (q * 5 + 0) * 2 + 0] = quad_1[q];
        host_dipoles[quad_start_offset + (q * 5 + 0) * 2 + 1] = 0.0;
        host_dipoles[quad_start_offset + (q * 5 + 1) * 2 + 0] = quad_2[q];
        host_dipoles[quad_start_offset + (q * 5 + 1) * 2 + 1] = 0.0;
        host_dipoles[quad_start_offset + (q * 5 + 2) * 2 + 0] = quad_3[q];
        host_dipoles[quad_start_offset + (q * 5 + 2) * 2 + 1] = 0.0;
        host_dipoles[quad_start_offset + (q * 5 + 3) * 2 + 0] = quad_4[q];
        host_dipoles[quad_start_offset + (q * 5 + 3) * 2 + 1] = 0.0;
        host_dipoles[quad_start_offset + (q * 5 + 4) * 2 + 0] = quad_5[q];
        host_dipoles[quad_start_offset + (q * 5 + 4) * 2 + 1] = 0.0;
    }
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dipoles.data(), (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyHostToDevice));
    dipoles_updated = true;
}

void Ewald_Electric_Field_Base::updateQuadrupolesComplex(const std::vector<double>& quad_1r, const std::vector<double>& quad_1i,
                                                        const std::vector<double>& quad_2r, const std::vector<double>& quad_2i,
                                                        const std::vector<double>& quad_3r, const std::vector<double>& quad_3i,
                                                        const std::vector<double>& quad_4r, const std::vector<double>& quad_4i,
                                                        const std::vector<double>& quad_5r, const std::vector<double>& quad_5i) {
    if (!solve_quadrupoles || num_quads == 0) {
        throw std::runtime_error("updateQuadrupolesComplex: Quadrupoles are not enabled for this solver.");
    }
    if (quad_1r.size() != num_quads || quad_1i.size() != num_quads ||
        quad_2r.size() != num_quads || quad_2i.size() != num_quads ||
        quad_3r.size() != num_quads || quad_3i.size() != num_quads ||
        quad_4r.size() != num_quads || quad_4i.size() != num_quads ||
        quad_5r.size() != num_quads || quad_5i.size() != num_quads) {
        throw std::invalid_argument("Input quadrupole component vectors must have size matching num_quads.");
    }
    std::vector<double> host_dipoles((num_particles * 3 + num_quads * 5) * 2, 0.0);
    if (d_dipoles != nullptr) {
        CUDA_CHECK(cudaMemcpy(host_dipoles.data(), d_dipoles, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    }
    size_t quad_start_offset = num_particles * 3 * 2;
    for (size_t q = 0; q < num_quads; ++q) {
        host_dipoles[quad_start_offset + (q * 5 + 0) * 2 + 0] = quad_1r[q];
        host_dipoles[quad_start_offset + (q * 5 + 0) * 2 + 1] = quad_1i[q];
        host_dipoles[quad_start_offset + (q * 5 + 1) * 2 + 0] = quad_2r[q];
        host_dipoles[quad_start_offset + (q * 5 + 1) * 2 + 1] = quad_2i[q];
        host_dipoles[quad_start_offset + (q * 5 + 2) * 2 + 0] = quad_3r[q];
        host_dipoles[quad_start_offset + (q * 5 + 2) * 2 + 1] = quad_3i[q];
        host_dipoles[quad_start_offset + (q * 5 + 3) * 2 + 0] = quad_4r[q];
        host_dipoles[quad_start_offset + (q * 5 + 3) * 2 + 1] = quad_4i[q];
        host_dipoles[quad_start_offset + (q * 5 + 4) * 2 + 0] = quad_5r[q];
        host_dipoles[quad_start_offset + (q * 5 + 4) * 2 + 1] = quad_5i[q];
    }
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dipoles.data(), (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyHostToDevice));
    dipoles_updated = true;
}

void Ewald_Electric_Field_Base::setSelfCoef(const std::vector<double>& self_coef_r, const std::vector<double>& self_coef_i) {
    if (self_coef_r.size() != num_particles || self_coef_i.size() != num_particles) {
        throw std::invalid_argument("Input self coefficients must have size matching num_particles.");
    }
    if (d_self_coef_r == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_self_coef_r, num_particles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, num_particles * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(d_self_coef_r, self_coef_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, self_coef_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
}

void Ewald_Electric_Field_Base::setSelfCoef(double val_r, double val_i) {
    if (d_self_coef_r == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_self_coef_r, num_particles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, num_particles * sizeof(double)));
    }
    std::vector<double> host_sc_r(num_particles, val_r);
    std::vector<double> host_sc_i(num_particles, val_i);
    CUDA_CHECK(cudaMemcpy(d_self_coef_r, host_sc_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, host_sc_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
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

void Ewald_Electric_Field_Base::getDipolesHost(std::vector<double>& host_dip_x,
                                              std::vector<double>& host_dip_y,
                                              std::vector<double>& host_dip_z) const {
    std::vector<double> host_dipoles((num_particles * 3 + num_quads * 5) * 2);
    CUDA_CHECK(cudaMemcpy(host_dipoles.data(), d_dipoles, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    host_dip_x.resize(num_particles);
    host_dip_y.resize(num_particles);
    host_dip_z.resize(num_particles);
    for (size_t i = 0; i < num_particles; ++i) {
        host_dip_x[i] = host_dipoles[(i * 3 + 0) * 2 + 0];
        host_dip_y[i] = host_dipoles[(i * 3 + 1) * 2 + 0];
        host_dip_z[i] = host_dipoles[(i * 3 + 2) * 2 + 0];
    }
}

void Ewald_Electric_Field_Base::getDipolesComplexHost(std::vector<double>& host_dip_xr, std::vector<double>& host_dip_xi,
                                                     std::vector<double>& host_dip_yr, std::vector<double>& host_dip_yi,
                                                     std::vector<double>& host_dip_zr, std::vector<double>& host_dip_zi) const {
    std::vector<double> host_dipoles((num_particles * 3 + num_quads * 5) * 2);
    CUDA_CHECK(cudaMemcpy(host_dipoles.data(), d_dipoles, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    host_dip_xr.resize(num_particles); host_dip_xi.resize(num_particles);
    host_dip_yr.resize(num_particles); host_dip_yi.resize(num_particles);
    host_dip_zr.resize(num_particles); host_dip_zi.resize(num_particles);
    for (size_t i = 0; i < num_particles; ++i) {
        host_dip_xr[i] = host_dipoles[(i * 3 + 0) * 2 + 0];
        host_dip_xi[i] = host_dipoles[(i * 3 + 0) * 2 + 1];
        host_dip_yr[i] = host_dipoles[(i * 3 + 1) * 2 + 0];
        host_dip_yi[i] = host_dipoles[(i * 3 + 1) * 2 + 1];
        host_dip_zr[i] = host_dipoles[(i * 3 + 2) * 2 + 0];
        host_dip_zi[i] = host_dipoles[(i * 3 + 2) * 2 + 1];
    }
}

void Ewald_Electric_Field_Base::getSpreadPrecalcsHost(std::vector<double>& host_spread_coef,
                                                     std::vector<int>& host_spread_idxs) const {
    host_spread_coef.resize(num_spread);
    host_spread_idxs.resize(num_spread);
    CUDA_CHECK(cudaMemcpy(host_spread_coef.data(), d_spread_coef, num_spread * sizeof(double), cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaMemcpy(host_contract_coef.data(), d_contract_coef, num_contract * sizeof(double), cudaMemcpyDeviceToHost));
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

std::vector<Complex> Ewald_Electric_Field_Base::getEPointHost() const {
    size_t num_targets = num_field_points > 0 ? num_field_points : num_particles;
    std::vector<Complex> host_E(num_targets * 3 + num_quads * 5, 0.0);
    if (d_E_point == nullptr || num_targets == 0) {
        return host_E;
    }
    std::vector<double> temp((num_targets * 3 + num_quads * 5) * 2);
    CUDA_CHECK(cudaMemcpy(temp.data(), d_E_point, (num_targets * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < num_targets * 3 + num_quads * 5; ++i) {
        host_E[i] = Complex(temp[i * 2], temp[i * 2 + 1]);
    }
    return host_E;
}

void Ewald_Electric_Field_Base::clearEPoint() {
    if (d_E_point) {
        size_t num_targets = num_field_points > 0 ? num_field_points : num_particles;
        size_t size_epoint_bytes = (num_targets * 3 + num_quads * 5) * 2 * sizeof(double);
        CUDA_CHECK(cudaMemset(d_E_point, 0, size_epoint_bytes));
    }
}
