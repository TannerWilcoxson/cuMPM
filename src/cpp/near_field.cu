#include "near_field.h"
#include "electric_field/ewald_electric_field.h"
#include "electric_field/polydisperse_electric_field.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
      if (abort) exit(code);
   }
}

Near_Field::Near_Field(const std::vector<double>& box,
                       const std::vector<Complex>& E0,
                       const std::vector<double>& radius,
                       double xi,
                       double errortol,
                       const std::string& field_type)
    : box(box), E0(E0), radius(radius), xi(xi), errortol(errortol), field_type(field_type) {
    if (box.size() != 3) {
        throw std::invalid_argument("Near_Field box must have length 3.");
    }
    if (E0.size() != 3) {
        throw std::invalid_argument("Near_Field E0 must have length 3.");
    }
}

Near_Field::~Near_Field() {}

void Near_Field::set_dipoles(const std::vector<Complex>& dip) {
    if (dip.size() % 3 != 0) {
        throw std::invalid_argument("Dipoles vector size must be a multiple of 3.");
    }
    num_particles = dip.size() / 3;

    dips_xr.resize(num_particles);
    dips_xi.resize(num_particles);
    dips_yr.resize(num_particles);
    dips_yi.resize(num_particles);
    dips_zr.resize(num_particles);
    dips_zi.resize(num_particles);

    for (size_t i = 0; i < num_particles; ++i) {
        dips_xr[i] = dip[i * 3 + 0].real();
        dips_xi[i] = dip[i * 3 + 0].imag();
        dips_yr[i] = dip[i * 3 + 1].real();
        dips_yi[i] = dip[i * 3 + 1].imag();
        dips_zr[i] = dip[i * 3 + 2].real();
        dips_zi[i] = dip[i * 3 + 2].imag();
    }
}

void Near_Field::set_dipole_positions(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Dipole coordinates x, y, z must have the same size.");
    }
    dip_pos_x = x;
    dip_pos_y = y;
    dip_pos_z = z;
}

void Near_Field::set_field_points(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z) {
    if (x.size() != y.size() || x.size() != z.size()) {
        throw std::invalid_argument("Field point coordinates x, y, z must have the same size.");
    }
    field_pos_x = x;
    field_pos_y = y;
    field_pos_z = z;
    num_field_points = x.size();
}

std::vector<double> Near_Field::calculate() {
    if (field_pos_x.empty() || dip_pos_x.empty() || dips_xr.empty()) {
        throw std::runtime_error("Near_Field: must set dipoles, dipole positions, and field points before calculation.");
    }

    if (dip_pos_x.size() != num_particles) {
        throw std::runtime_error("Near_Field: dipole position size is inconsistent with dipole vector size.");
    }

    if (!EF) {
        if (radius.empty()) {
            radius.resize(num_particles, 1.0);
        } else if (radius.size() == 1) {
            double val = radius[0];
            radius.resize(num_particles, val);
        } else if (radius.size() != num_particles) {
            throw std::runtime_error("The number of particles is inconsistent with the number of radii provided.");
        }

        length_scale = radius[0];

        double scaled_box_x = box[0] / length_scale;
        double scaled_box_y = box[1] / length_scale;
        double scaled_box_z = box[2] / length_scale;

        std::vector<double> scaled_radius = radius;
        for (size_t i = 0; i < scaled_radius.size(); ++i) {
            scaled_radius[i] /= length_scale;
        }

        use_polydisperse = false;
        if (field_type == "polydisperse") {
            use_polydisperse = true;
        } else if (field_type == "monodisperse") {
            use_polydisperse = false;
        } else if (field_type == "auto") {
            bool is_monodisperse = true;
            for (size_t i = 1; i < num_particles; ++i) {
                if (radius[i] != radius[0]) {
                    is_monodisperse = false;
                    break;
                }
            }
            use_polydisperse = !is_monodisperse;
        } else {
            throw std::runtime_error("Unknown field_type: " + field_type);
        }

        if (use_polydisperse) {
            EF = std::make_unique<Polydisperse_Electric_Field>(
                scaled_box_x, scaled_box_y, scaled_box_z, errortol, xi, false, scaled_radius
            );
        } else {
            EF = std::make_unique<Ewald_Electric_Field>(
                scaled_box_x, scaled_box_y, scaled_box_z, errortol, xi, false
            );
        }
    }

    // Update coordinate buffers and scale them by length_scale
    std::vector<double> scaled_dip_x(num_particles);
    std::vector<double> scaled_dip_y(num_particles);
    std::vector<double> scaled_dip_z(num_particles);
    for (size_t i = 0; i < num_particles; ++i) {
        scaled_dip_x[i] = dip_pos_x[i] / length_scale;
        scaled_dip_y[i] = dip_pos_y[i] / length_scale;
        scaled_dip_z[i] = dip_pos_z[i] / length_scale;
    }
    EF->updateParticleCoordinates(scaled_dip_x, scaled_dip_y, scaled_dip_z);

    std::vector<double> scaled_field_x(num_field_points);
    std::vector<double> scaled_field_y(num_field_points);
    std::vector<double> scaled_field_z(num_field_points);
    for (size_t i = 0; i < num_field_points; ++i) {
        scaled_field_x[i] = field_pos_x[i] / length_scale;
        scaled_field_y[i] = field_pos_y[i] / length_scale;
        scaled_field_z[i] = field_pos_z[i] / length_scale;
    }

    if (use_polydisperse) {
        auto* derived = static_cast<Polydisperse_Electric_Field*>(EF.get());
        derived->updateFieldCoordinates(scaled_field_x, scaled_field_y, scaled_field_z);
        derived->updateDipolesComplex(dips_xr, dips_xi, dips_yr, dips_yi, dips_zr, dips_zi);
        // set self coeff to 0 (near field has no self-interaction for arbitrary points)
        std::vector<double> self_r(num_particles, 0.0);
        std::vector<double> self_i(num_particles, 0.0);
        derived->setSelfCoef(self_r, self_i);
    } else {
        auto* derived = static_cast<Ewald_Electric_Field*>(EF.get());
        derived->updateFieldCoordinates(scaled_field_x, scaled_field_y, scaled_field_z);
        derived->updateDipolesComplex(dips_xr, dips_xi, dips_yr, dips_yi, dips_zr, dips_zi);
        // set self coeff to 0
        std::vector<double> self_r(num_particles, 0.0);
        std::vector<double> self_i(num_particles, 0.0);
        derived->setSelfCoef(self_r, self_i);
    }

    // Run GPU calculations
    EF->calculate();

    // Copy field vectors from GPU (size: num_field_points * 3 * 2)
    std::vector<double> host_E_point(num_field_points * 3 * 2);
    CUDA_CHECK(cudaMemcpy(host_E_point.data(), EF->getDevEPoint(), num_field_points * 3 * 2 * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute intensity: | -E_ind + E0 |^2
    std::vector<double> intensity(num_field_points, 0.0);
    for (size_t i = 0; i < num_field_points; ++i) {
        double fx_r = -host_E_point[(i * 3 + 0) * 2 + 0] + E0[0].real();
        double fx_i = -host_E_point[(i * 3 + 0) * 2 + 1] + E0[0].imag();

        double fy_r = -host_E_point[(i * 3 + 1) * 2 + 0] + E0[1].real();
        double fy_i = -host_E_point[(i * 3 + 1) * 2 + 1] + E0[1].imag();

        double fz_r = -host_E_point[(i * 3 + 2) * 2 + 0] + E0[2].real();
        double fz_i = -host_E_point[(i * 3 + 2) * 2 + 1] + E0[2].imag();

        intensity[i] = (fx_r * fx_r + fx_i * fx_i) +
                       (fy_r * fy_r + fy_i * fy_i) +
                       (fz_r * fz_r + fz_i * fz_i);
    }

    return intensity;
}
