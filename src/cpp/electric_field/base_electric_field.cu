#include "electric_field.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// CUDA Error Checking macro
#ifndef CUDA_CHECK
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
      if (abort) exit(code);
   }
}
#endif

Base_Electric_Field::Base_Electric_Field(FieldCalcMode mode,
                                         bool solve_quadrupoles,
                                         const std::vector<int>& quad_idxs)
    : mode(mode), solve_quadrupoles(solve_quadrupoles), quad_idxs(quad_idxs) {}

Base_Electric_Field::~Base_Electric_Field() {
    if (d_x_field == d_x_part) d_x_field = nullptr;
    if (d_y_field == d_y_part) d_y_field = nullptr;
    if (d_z_field == d_z_part) d_z_field = nullptr;

    if (d_x_part)      { cudaFree(d_x_part);      d_x_part      = nullptr; }
    if (d_y_part)      { cudaFree(d_y_part);      d_y_part      = nullptr; }
    if (d_z_part)      { cudaFree(d_z_part);      d_z_part      = nullptr; }
    if (d_dipoles)     { cudaFree(d_dipoles);     d_dipoles     = nullptr; }
    if (d_self_coef_r) { cudaFree(d_self_coef_r); d_self_coef_r = nullptr; }
    if (d_self_coef_i) { cudaFree(d_self_coef_i); d_self_coef_i = nullptr; }
    if (d_x_field)     { cudaFree(d_x_field);     d_x_field     = nullptr; }
    if (d_y_field)     { cudaFree(d_y_field);     d_y_field     = nullptr; }
    if (d_z_field)     { cudaFree(d_z_field);     d_z_field     = nullptr; }
    if (d_E_point)     { cudaFree(d_E_point);     d_E_point     = nullptr; }
    if (d_quad_idxs)   { cudaFree(d_quad_idxs);   d_quad_idxs   = nullptr; }
    if (d_quad_map)    { cudaFree(d_quad_map);    d_quad_map    = nullptr; }
}

void Base_Electric_Field::updateParticleCoordinates(
    const std::vector<double>& x_part,
    const std::vector<double>& y_part,
    const std::vector<double>& z_part)
{
    if (x_part.size() != y_part.size() || x_part.size() != z_part.size()) {
        throw std::invalid_argument("Base_Electric_Field: coordinate vectors must have equal size.");
    }
    size_t N = x_part.size();

    if (d_quad_idxs) { cudaFree(d_quad_idxs); d_quad_idxs = nullptr; }
    if (d_quad_map)  { cudaFree(d_quad_map);  d_quad_map  = nullptr; }

    if (solve_quadrupoles) {
        num_quads = quad_idxs.empty() ? N : quad_idxs.size();
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

        CUDA_CHECK(cudaMalloc(&d_quad_map, N * sizeof(int)));
        std::vector<int> host_quad_map(N, -1);
        for (size_t q = 0; q < num_quads; ++q) {
            int p_idx = host_quad_idxs[q];
            if (p_idx >= 0 && p_idx < static_cast<int>(N)) {
                host_quad_map[p_idx] = q;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_quad_map, host_quad_map.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        num_quads = 0;
    }

    if (N != num_particles || d_x_part == nullptr) {
        if (d_x_part)      { cudaFree(d_x_part);      d_x_part      = nullptr; }
        if (d_y_part)      { cudaFree(d_y_part);      d_y_part      = nullptr; }
        if (d_z_part)      { cudaFree(d_z_part);      d_z_part      = nullptr; }
        if (d_dipoles)     { cudaFree(d_dipoles);     d_dipoles     = nullptr; }
        if (d_self_coef_r) { cudaFree(d_self_coef_r); d_self_coef_r = nullptr; }
        if (d_self_coef_i) { cudaFree(d_self_coef_i); d_self_coef_i = nullptr; }
        if (d_E_point)     { cudaFree(d_E_point);     d_E_point     = nullptr; }

        num_particles = N;
        if (N > 0) {
            CUDA_CHECK(cudaMalloc(&d_x_part, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_y_part, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_z_part, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_dipoles, (N * 3 + num_quads * 5) * 2 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_dipoles, 0, (N * 3 + num_quads * 5) * 2 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_self_coef_r, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_self_coef_i, N * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_self_coef_r, 0, N * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_self_coef_i, 0, N * sizeof(double)));

            size_t num_targets = num_field_points > 0 ? num_field_points : N;
            CUDA_CHECK(cudaMalloc(&d_E_point, (num_targets * 3 + num_quads * 5) * 2 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_E_point, 0, (num_targets * 3 + num_quads * 5) * 2 * sizeof(double)));
        }
    }

    if (N > 0) {
        CUDA_CHECK(cudaMemcpy(d_x_part, x_part.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_part, y_part.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z_part, z_part.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    }

    particles_updated = true;

    if (mode == FieldCalcMode::SOLVER_AX) {
        d_x_field = d_x_part;
        d_y_field = d_y_part;
        d_z_field = d_z_part;
        num_field_points = num_particles;
        field_points_updated = true;
    }
}

void Base_Electric_Field::updateFieldCoordinates(
    const std::vector<double>& x_field,
    const std::vector<double>& y_field,
    const std::vector<double>& z_field)
{
    if (mode == FieldCalcMode::SOLVER_AX) {
        throw std::runtime_error("Base_Electric_Field: Cannot update field coordinates when SOLVER_AX mode is active.");
    }
    if (x_field.size() != y_field.size() || x_field.size() != z_field.size()) {
        throw std::invalid_argument("Base_Electric_Field: field coordinate vectors must have equal size.");
    }
    size_t M = x_field.size();
    if (M == 0) {
        if (d_x_field)     { cudaFree(d_x_field);     d_x_field     = nullptr; }
        if (d_y_field)     { cudaFree(d_y_field);     d_y_field     = nullptr; }
        if (d_z_field)     { cudaFree(d_z_field);     d_z_field     = nullptr; }
        if (d_E_point)     { cudaFree(d_E_point);     d_E_point     = nullptr; }
        num_field_points = 0;
        field_points_updated = false;
        
        if (num_particles > 0) {
            CUDA_CHECK(cudaMalloc(&d_E_point, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_E_point, 0, (num_particles * 3 + num_quads * 5) * 2 * sizeof(double)));
        }
        return;
    }

    if (M != num_field_points) {
        if (d_x_field) { cudaFree(d_x_field); d_x_field = nullptr; }
        if (d_y_field) { cudaFree(d_y_field); d_y_field = nullptr; }
        if (d_z_field) { cudaFree(d_z_field); d_z_field = nullptr; }
        if (d_E_point) { cudaFree(d_E_point); d_E_point = nullptr; }

        num_field_points = M;
        CUDA_CHECK(cudaMalloc(&d_x_field, M * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y_field, M * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z_field, M * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_E_point, (M * 3 + num_quads * 5) * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_E_point, 0, (M * 3 + num_quads * 5) * 2 * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(d_x_field, x_field.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_field, y_field.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_field, z_field.data(), M * sizeof(double), cudaMemcpyHostToDevice));

    field_points_updated = true;
}

void Base_Electric_Field::updateDipoles(const std::vector<double>& dip_x,
                                       const std::vector<double>& dip_y,
                                       const std::vector<double>& dip_z)
{
    if (dip_x.size() != num_particles || dip_y.size() != num_particles || dip_z.size() != num_particles) {
        throw std::invalid_argument("Base_Electric_Field: input dipole vectors must match the allocated number of particles.");
    }
    if (num_particles == 0) return;

    std::vector<double> host_dips(num_particles * 3 * 2, 0.0);
    for (size_t p = 0; p < num_particles; ++p) {
        host_dips[(p * 3 + 0) * 2 + 0] = dip_x[p];
        host_dips[(p * 3 + 1) * 2 + 0] = dip_y[p];
        host_dips[(p * 3 + 2) * 2 + 0] = dip_z[p];
    }

    size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dips.data(), size_dipoles_bytes, cudaMemcpyHostToDevice));

    dipoles_updated = true;
}

void Base_Electric_Field::updateDipolesComplex(
    const std::vector<double>& dip_xr, const std::vector<double>& dip_xi,
    const std::vector<double>& dip_yr, const std::vector<double>& dip_yi,
    const std::vector<double>& dip_zr, const std::vector<double>& dip_zi)
{
    if (dip_xr.size() != num_particles || dip_xi.size() != num_particles ||
        dip_yr.size() != num_particles || dip_yi.size() != num_particles ||
        dip_zr.size() != num_particles || dip_zi.size() != num_particles) {
        throw std::invalid_argument("Base_Electric_Field: complex dipole component vectors must match the allocated number of particles.");
    }
    if (num_particles == 0) return;

    std::vector<double> host_dips(num_particles * 3 * 2);
    for (size_t p = 0; p < num_particles; ++p) {
        host_dips[(p * 3 + 0) * 2 + 0] = dip_xr[p];
        host_dips[(p * 3 + 0) * 2 + 1] = dip_xi[p];
        host_dips[(p * 3 + 1) * 2 + 0] = dip_yr[p];
        host_dips[(p * 3 + 1) * 2 + 1] = dip_yi[p];
        host_dips[(p * 3 + 2) * 2 + 0] = dip_zr[p];
        host_dips[(p * 3 + 2) * 2 + 1] = dip_zi[p];
    }

    size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_dipoles, host_dips.data(), size_dipoles_bytes, cudaMemcpyHostToDevice));

    dipoles_updated = true;
}

void Base_Electric_Field::updateQuadrupoles(
    const std::vector<double>& quad_1, const std::vector<double>& quad_2,
    const std::vector<double>& quad_3, const std::vector<double>& quad_4,
    const std::vector<double>& quad_5)
{
    if (!solve_quadrupoles) return;
    if (quad_1.size() != num_quads || quad_2.size() != num_quads ||
        quad_3.size() != num_quads || quad_4.size() != num_quads ||
        quad_5.size() != num_quads) {
        throw std::invalid_argument("Base_Electric_Field: quadrupole component vectors must match num_quads.");
    }
    if (num_quads == 0) return;

    std::vector<double> host_quads(num_quads * 5 * 2, 0.0);
    for (size_t q = 0; q < num_quads; ++q) {
        host_quads[(q * 5 + 0) * 2 + 0] = quad_1[q];
        host_quads[(q * 5 + 1) * 2 + 0] = quad_2[q];
        host_quads[(q * 5 + 2) * 2 + 0] = quad_3[q];
        host_quads[(q * 5 + 3) * 2 + 0] = quad_4[q];
        host_quads[(q * 5 + 4) * 2 + 0] = quad_5[q];
    }

    size_t offset = num_particles * 3 * 2;
    size_t size_quads_bytes = num_quads * 5 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_dipoles + offset, host_quads.data(), size_quads_bytes, cudaMemcpyHostToDevice));

    dipoles_updated = true;
}

void Base_Electric_Field::updateQuadrupolesComplex(
    const std::vector<double>& quad_1r, const std::vector<double>& quad_1i,
    const std::vector<double>& quad_2r, const std::vector<double>& quad_2i,
    const std::vector<double>& quad_3r, const std::vector<double>& quad_3i,
    const std::vector<double>& quad_4r, const std::vector<double>& quad_4i,
    const std::vector<double>& quad_5r, const std::vector<double>& quad_5i)
{
    if (!solve_quadrupoles) return;
    if (quad_1r.size() != num_quads || quad_1i.size() != num_quads ||
        quad_2r.size() != num_quads || quad_2i.size() != num_quads ||
        quad_3r.size() != num_quads || quad_3i.size() != num_quads ||
        quad_4r.size() != num_quads || quad_4i.size() != num_quads ||
        quad_5r.size() != num_quads || quad_5i.size() != num_quads) {
        throw std::invalid_argument("Base_Electric_Field: complex quadrupole component vectors must match num_quads.");
    }
    if (num_quads == 0) return;

    std::vector<double> host_quads(num_quads * 5 * 2, 0.0);
    for (size_t q = 0; q < num_quads; ++q) {
        host_quads[(q * 5 + 0) * 2 + 0] = quad_1r[q];
        host_quads[(q * 5 + 0) * 2 + 1] = quad_1i[q];
        host_quads[(q * 5 + 1) * 2 + 0] = quad_2r[q];
        host_quads[(q * 5 + 1) * 2 + 1] = quad_2i[q];
        host_quads[(q * 5 + 2) * 2 + 0] = quad_3r[q];
        host_quads[(q * 5 + 2) * 2 + 1] = quad_3i[q];
        host_quads[(q * 5 + 3) * 2 + 0] = quad_4r[q];
        host_quads[(q * 5 + 3) * 2 + 1] = quad_4i[q];
        host_quads[(q * 5 + 4) * 2 + 0] = quad_5r[q];
        host_quads[(q * 5 + 4) * 2 + 1] = quad_5i[q];
    }

    size_t offset = num_particles * 3 * 2;
    size_t size_quads_bytes = num_quads * 5 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_dipoles + offset, host_quads.data(), size_quads_bytes, cudaMemcpyHostToDevice));

    dipoles_updated = true;
}

void Base_Electric_Field::getDipolesHost(std::vector<double>& host_dip_x,
                                        std::vector<double>& host_dip_y,
                                        std::vector<double>& host_dip_z) const
{
    if (num_particles == 0 || d_dipoles == nullptr) {
        throw std::runtime_error("Base_Electric_Field: dipoles have not been allocated on GPU yet.");
    }

    host_dip_x.resize(num_particles);
    host_dip_y.resize(num_particles);
    host_dip_z.resize(num_particles);

    std::vector<double> host_dips(num_particles * 3 * 2);
    size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(host_dips.data(), d_dipoles, size_dipoles_bytes, cudaMemcpyDeviceToHost));

    for (size_t p = 0; p < num_particles; ++p) {
        host_dip_x[p] = host_dips[(p * 3 + 0) * 2 + 0];
        host_dip_y[p] = host_dips[(p * 3 + 1) * 2 + 0];
        host_dip_z[p] = host_dips[(p * 3 + 2) * 2 + 0];
    }
}

void Base_Electric_Field::getDipolesComplexHost(
    std::vector<double>& host_dip_xr, std::vector<double>& host_dip_xi,
    std::vector<double>& host_dip_yr, std::vector<double>& host_dip_yi,
    std::vector<double>& host_dip_zr, std::vector<double>& host_dip_zi) const
{
    if (num_particles == 0 || d_dipoles == nullptr) {
        throw std::runtime_error("Base_Electric_Field: dipoles have not been allocated on GPU yet.");
    }

    host_dip_xr.resize(num_particles);
    host_dip_xi.resize(num_particles);
    host_dip_yr.resize(num_particles);
    host_dip_yi.resize(num_particles);
    host_dip_zr.resize(num_particles);
    host_dip_zi.resize(num_particles);

    std::vector<double> host_dips(num_particles * 3 * 2);
    size_t size_dipoles_bytes = num_particles * 3 * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(host_dips.data(), d_dipoles, size_dipoles_bytes, cudaMemcpyDeviceToHost));

    for (size_t p = 0; p < num_particles; ++p) {
        host_dip_xr[p] = host_dips[(p * 3 + 0) * 2 + 0];
        host_dip_xi[p] = host_dips[(p * 3 + 0) * 2 + 1];
        host_dip_yr[p] = host_dips[(p * 3 + 1) * 2 + 0];
        host_dip_yi[p] = host_dips[(p * 3 + 1) * 2 + 1];
        host_dip_zr[p] = host_dips[(p * 3 + 2) * 2 + 0];
        host_dip_zi[p] = host_dips[(p * 3 + 2) * 2 + 1];
    }
}

void Base_Electric_Field::setSelfCoef(
    const std::vector<double>& self_coef_r,
    const std::vector<double>& self_coef_i)
{
    if (self_coef_r.size() != num_particles || self_coef_i.size() != num_particles) {
        throw std::invalid_argument("Base_Electric_Field: self_coef size must equal num_particles.");
    }
    if (d_self_coef_r == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_self_coef_r, num_particles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, num_particles * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(d_self_coef_r, self_coef_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, self_coef_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
}

void Base_Electric_Field::setSelfCoef(double val_r, double val_i) {
    if (d_self_coef_r == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_self_coef_r, num_particles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_self_coef_i, num_particles * sizeof(double)));
    }
    std::vector<double> host_sc_r(num_particles, val_r);
    std::vector<double> host_sc_i(num_particles, val_i);
    CUDA_CHECK(cudaMemcpy(d_self_coef_r, host_sc_r.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self_coef_i, host_sc_i.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
}

std::vector<std::complex<double>> Base_Electric_Field::getEPointHost() const {
    size_t num_targets = num_field_points > 0 ? num_field_points : num_particles;
    std::vector<std::complex<double>> host_E(num_targets * 3 + num_quads * 5, 0.0);
    if (d_E_point == nullptr || num_targets == 0) {
        return host_E;
    }
    std::vector<double> temp((num_targets * 3 + num_quads * 5) * 2);
    CUDA_CHECK(cudaMemcpy(temp.data(), d_E_point, (num_targets * 3 + num_quads * 5) * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < num_targets * 3 + num_quads * 5; ++i) {
        host_E[i] = std::complex<double>(temp[i * 2], temp[i * 2 + 1]);
    }
    return host_E;
}

void Base_Electric_Field::clearEPoint() {
    if (d_E_point) {
        size_t num_targets = num_field_points > 0 ? num_field_points : num_particles;
        size_t size_epoint_bytes = (num_targets * 3 + num_quads * 5) * 2 * sizeof(double);
        CUDA_CHECK(cudaMemset(d_E_point, 0, size_epoint_bytes));
    }
}
