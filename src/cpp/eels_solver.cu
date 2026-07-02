#include "eels_solver.h"
#include "electric_field/monodisperse_ewald_electric_field.h"
#include "electric_field/polydisperse_ewald_electric_field.h"
#include "electric_field/direct_electric_field.h"
#include "numerical_solver/gmres_solver.h"
#include "numerical_solver/bicgstab_solver.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(ans) { gpuAssert_eels((ans), __FILE__, __LINE__); }
inline void gpuAssert_eels(cudaError_t code, const char *file, int line, bool abort = true) {
   if (code != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
      if (abort) exit(code);
   }
}

// Helper functions for modified Bessel functions of complex arguments z = x + i*y
// designed for purely real (y=0) and purely imaginary (x=0) inputs.
static Complex eval_k0(Complex z) {
    if (std::abs(z.imag()) < 1e-12) {
        return std::cyl_bessel_k(0, z.real());
    } else if (std::abs(z.real()) < 1e-12) {
        double y = z.imag();
        if (y < 0.0) {
            double j0 = std::cyl_bessel_j(0, -y);
            double y0 = std::cyl_neumann(0, -y);
            return std::conj(Complex(-M_PI / 2.0 * y0, -M_PI / 2.0 * j0));
        } else {
            double j0 = std::cyl_bessel_j(0, y);
            double y0 = std::cyl_neumann(0, y);
            return Complex(-M_PI / 2.0 * y0, -M_PI / 2.0 * j0);
        }
    } else {
        return std::cyl_bessel_k(0, z.real());
    }
}

static Complex eval_k1(Complex z) {
    if (std::abs(z.imag()) < 1e-12) {
        return std::cyl_bessel_k(1, z.real());
    } else if (std::abs(z.real()) < 1e-12) {
        double y = z.imag();
        if (y < 0.0) {
            double j1 = std::cyl_bessel_j(1, -y);
            double y1 = std::cyl_neumann(1, -y);
            return std::conj(Complex(-M_PI / 2.0 * j1, M_PI / 2.0 * y1));
        } else {
            double j1 = std::cyl_bessel_j(1, y);
            double y1 = std::cyl_neumann(1, y);
            return Complex(-M_PI / 2.0 * j1, M_PI / 2.0 * y1);
        }
    } else {
        return std::cyl_bessel_k(1, z.real());
    }
}

// Simpson's integration helper
static double simpson_integrate(const std::vector<double>& f, double h) {
    size_t n = f.size();
    if (n < 2) return 0.0;
    if (n == 2) {
        return 0.5 * h * (f[0] + f[1]);
    }
    
    double sum = 0.0;
    if (n % 2 == 1) {
        sum += f[0] + f[n - 1];
        for (size_t i = 1; i < n - 1; ++i) {
            sum += (i % 2 == 1 ? 4.0 : 2.0) * f[i];
        }
        return sum * h / 3.0;
    } else {
        sum += f[0] + f[n - 2];
        for (size_t i = 1; i < n - 2; ++i) {
            sum += (i % 2 == 1 ? 4.0 : 2.0) * f[i];
        }
        double simp_part = sum * h / 3.0;
        double trap_part = 0.5 * h * (f[n - 2] + f[n - 1]);
        return simp_part + trap_part;
    }
}

// Constructor 1: Full 2D eps_p
EELS_Solver::EELS_Solver(const std::vector<double>& box,
                         const std::vector<std::vector<Complex>>& eps_p,
                         const std::vector<double>& omega,
                         double v,
                         const std::vector<double>& radius,
                         double eps_m,
                         double xi,
                         double tol,
                         bool quiet,
                         const std::string& guess_type,
                         const std::string& solver_type,
                         const std::string& field_type,
                         double integration_step,
                         bool solve_quadrupoles,
                         const std::vector<int>& quad_idxs,
                         bool asm_flag,
                         int asm_Nx,
                         int asm_Ny)
    : box(box), eps_p(eps_p), omega(omega), v(v), radius(radius), eps_m(eps_m), xi(xi), tol(tol), quiet(quiet), guess_type(guess_type), solver_type(solver_type), field_type(field_type), integration_step(integration_step), solve_quadrupoles(solve_quadrupoles), quad_idxs(quad_idxs), asm_flag(asm_flag), asm_Nx(asm_Nx), asm_Ny(asm_Ny) {}

// Constructor 2: 1D eps_p
EELS_Solver::EELS_Solver(const std::vector<double>& box,
                         const std::vector<Complex>& eps_p_1d,
                         const std::vector<double>& omega,
                         double v,
                         const std::vector<double>& radius,
                         double eps_m,
                         double xi,
                         double tol,
                         bool quiet,
                         const std::string& guess_type,
                         const std::string& solver_type,
                         const std::string& field_type,
                         double integration_step,
                         bool solve_quadrupoles,
                         const std::vector<int>& quad_idxs,
                         bool asm_flag,
                         int asm_Nx,
                         int asm_Ny)
    : box(box), omega(omega), v(v), radius(radius), eps_m(eps_m), xi(xi), tol(tol), quiet(quiet), guess_type(guess_type), solver_type(solver_type), field_type(field_type), integration_step(integration_step), solve_quadrupoles(solve_quadrupoles), quad_idxs(quad_idxs), asm_flag(asm_flag), asm_Nx(asm_Nx), asm_Ny(asm_Ny) {
    this->eps_p.resize(eps_p_1d.size(), std::vector<Complex>(1, 0.0));
    for (size_t w = 0; w < eps_p_1d.size(); ++w) {
        this->eps_p[w][0] = eps_p_1d[w];
    }
}

// Constructor 3: Scalar eps_p
EELS_Solver::EELS_Solver(const std::vector<double>& box,
                         Complex eps_p_scalar,
                         const std::vector<double>& omega,
                         double v,
                         const std::vector<double>& radius,
                         double eps_m,
                         double xi,
                         double tol,
                         bool quiet,
                         const std::string& guess_type,
                         const std::string& solver_type,
                         const std::string& field_type,
                         double integration_step,
                         bool solve_quadrupoles,
                         const std::vector<int>& quad_idxs,
                         bool asm_flag,
                         int asm_Nx,
                         int asm_Ny)
    : box(box), omega(omega), v(v), radius(radius), eps_m(eps_m), xi(xi), tol(tol), quiet(quiet), guess_type(guess_type), solver_type(solver_type), field_type(field_type), integration_step(integration_step), solve_quadrupoles(solve_quadrupoles), quad_idxs(quad_idxs), asm_flag(asm_flag), asm_Nx(asm_Nx), asm_Ny(asm_Ny) {
    this->eps_p.resize(1, std::vector<Complex>(1, eps_p_scalar));
}

EELS_Solver::~EELS_Solver() {}

void EELS_Solver::print(const std::string& msg) const {
    if (!quiet) {
        for (int i = 0; i < indent_level; ++i) std::cout << "  ";
        std::cout << msg << std::endl;
    }
}

void EELS_Solver::set_dims(size_t num_p) {
    num_particles = num_p;
    num_quads = solve_quadrupoles ? (quad_idxs.empty() ? num_particles : quad_idxs.size()) : 0;

    if (radius.empty()) {
        radius.resize(num_particles, 1.0);
    } else if (radius.size() == 1) {
        double val = radius[0];
        radius.resize(num_particles, val);
    } else if (radius.size() != num_particles) {
        throw std::runtime_error("The number of particles is inconsistent with the number of radii provided.");
    }

    bool is_monodisperse = true;
    for (size_t i = 1; i < num_particles; ++i) {
        if (radius[i] != radius[0]) {
            is_monodisperse = false;
            break;
        }
    }
    if (field_type == "monodisperse" && !is_monodisperse) {
        throw std::runtime_error("field_type was set to 'monodisperse' but radii are not identical.");
    }

    if (eps_p.empty()) {
        throw std::runtime_error("eps_p is empty.");
    }
    num_wavevectors = eps_p.size();

    for (size_t w = 0; w < num_wavevectors; ++w) {
        if (eps_p[w].size() == 1) {
            Complex val = eps_p[w][0];
            eps_p[w].resize(num_particles, val);
        } else if (eps_p[w].size() != num_particles) {
            throw std::runtime_error("The number of particles is inconsistent with eps_p provided.");
        }
    }
}

void EELS_Solver::nondimensionalize() {
    length_scale = *std::min_element(radius.begin(), radius.end());
    eps_scale = eps_m;

    box[0] /= length_scale;
    box[1] /= length_scale;
    box[2] /= length_scale;

    for (size_t i = 0; i < radius.size(); ++i) {
        radius[i] /= length_scale;
    }

    eps_m /= eps_scale; // becomes 1.0

    for (size_t w = 0; w < num_wavevectors; ++w) {
        omega[w] *= length_scale;
        for (size_t p = 0; p < num_particles; ++p) {
            eps_p[w][p] /= eps_scale;
        }
    }

    integration_step /= length_scale;
}

void EELS_Solver::precalculations() {
    double sum_r3 = 0.0;
    for (double r : radius) {
        sum_r3 += std::pow(r, 3.0);
    }
    double box_vol = box[0] * box[1] * box[2];
    double vol_frac = (4.0 / 3.0) * M_PI * sum_r3 / box_vol;
    (void)vol_frac; // unused for eels but matches precalc footprint
}

void EELS_Solver::precomp_eels(const std::vector<double>& z_part) {
    double dz = integration_step;
    bool any_neg = false;
    for (double zp : z_part) {
        if (zp <= 0.0) {
            any_neg = true;
            break;
        }
    }

    double zmin, zmax;
    double Z;
    if (any_neg) {
        zmin = -box[2] / 2.0;
        zmax = box[2] / 2.0 - dz;
        Z = 0.0;
    } else {
        zmin = 0.0;
        zmax = box[2] - dz;
        Z = 1.0;
    }

    double P0 = z_part[0];
    z_pts.clear();
    double cur_z = zmin;
    while (cur_z <= zmax + dz / 2.0) {
        z_pts.push_back(cur_z);
        cur_z += dz;
    }

    Z_pts.resize(z_pts.size());
    for (size_t i = 0; i < z_pts.size(); ++i) {
        double val = z_pts[i] - P0 + Z * box[2] / 2.0;
        if (val < zmin) {
            val += box[2];
        }
        if (val > zmax + dz / 2.0) {
            val -= box[2];
        }
        Z_pts[i] = val + P0 - Z * box[2] / 2.0;
    }
}

// Guess Predictor Calculations
std::vector<Complex> EELS_Solver::calc_guess(const std::vector<Complex>& prev_dip, const std::vector<Complex>& prev_quad, size_t wavevec_idx, const std::vector<Complex>& E_inc) const {
    if (guess_type == "mean_field" || guess_type == "mean-field") {
        return calc_mean_field_guess(wavevec_idx, E_inc);
    } else if (guess_type == "previous") {
        return calc_previous_guess(prev_dip, prev_quad, wavevec_idx);
    } else if (guess_type == "derivative") {
        return calc_derivative_guess(prev_dip, prev_quad, wavevec_idx);
    } else {
        throw std::runtime_error("Guess type " + guess_type + " not supported.");
    }
}

std::vector<Complex> EELS_Solver::calc_mean_field_guess(size_t wavevec_idx, const std::vector<Complex>& E_inc) const {
    std::vector<Complex> solver_guess(num_particles * 3 + num_quads * 5, 0.0);
    double sum_r3 = 0.0;
    for (double r : radius) {
        sum_r3 += std::pow(r, 3.0);
    }
    double box_vol = box[0] * box[1] * box[2];
    double vol_frac = (4.0 / 3.0) * M_PI * sum_r3 / box_vol;

    for (size_t p = 0; p < num_particles; ++p) {
        Complex beta = (eps_p[wavevec_idx][p] - 1.0) / (eps_p[wavevec_idx][p] + 2.0);
        Complex val = 4.0 * M_PI * beta / (1.0 - beta * vol_frac);
        for (int c = 0; c < 3; ++c) {
            solver_guess[p * 3 + c] = val * E_inc[p * 3 + c];
        }
    }
    // Quadrupoles remain 0.0 for mean field guess (already initialized to 0.0)
    return solver_guess;
}

std::vector<Complex> EELS_Solver::calc_previous_guess(const std::vector<Complex>& prev_dip, const std::vector<Complex>& prev_quad, size_t wavevec_idx) const {
    if (wavevec_idx < 1) {
        return std::vector<Complex>(num_particles * 3 + num_quads * 5, 0.0);
    }
    std::vector<Complex> solver_guess(num_particles * 3 + num_quads * 5);
    size_t dip_offset = (wavevec_idx - 1) * num_particles * 3;
    std::copy(prev_dip.begin() + dip_offset, prev_dip.begin() + dip_offset + num_particles * 3, solver_guess.begin());

    if (solve_quadrupoles && num_quads > 0) {
        size_t quad_offset = (wavevec_idx - 1) * num_quads * 5;
        std::copy(prev_quad.begin() + quad_offset, prev_quad.begin() + quad_offset + num_quads * 5, solver_guess.begin() + num_particles * 3);
    }
    return solver_guess;
}

std::vector<Complex> EELS_Solver::calc_derivative_guess(const std::vector<Complex>& prev_dip, const std::vector<Complex>& prev_quad, size_t wavevec_idx) const {
    if (wavevec_idx < 2) {
        return calc_previous_guess(prev_dip, prev_quad, wavevec_idx);
    }
    size_t i = wavevec_idx;
    size_t im2 = i - 2;
    size_t im1 = i - 1;
    std::vector<Complex> solver_guess(num_particles * 3 + num_quads * 5);

    // Predict dipoles
    for (size_t p = 0; p < num_particles; ++p) {
        Complex eps_im2 = eps_p[im2][p];
        Complex eps_im1 = eps_p[im1][p];
        Complex eps_i = eps_p[i][p];

        Complex X_im2 = (eps_im2 - 1.0 == 0.0) ? 0.0 : (eps_im2 + 2.0) / (eps_im2 - 1.0);
        Complex X_im1 = (eps_im1 - 1.0 == 0.0) ? 0.0 : (eps_im1 + 2.0) / (eps_im1 - 1.0);
        Complex X_i = (eps_i - 1.0 == 0.0) ? 0.0 : (eps_i + 2.0) / (eps_i - 1.0);

        Complex run = X_im1 - X_im2;
        Complex new_run = X_i - X_im1;
        if (run == 0.0) {
            run = 1.0;
            new_run = 0.0;
        }
        for (int c = 0; c < 3; ++c) {
            Complex val_im1 = prev_dip[im1 * num_particles * 3 + p * 3 + c];
            Complex val_im2 = prev_dip[im2 * num_particles * 3 + p * 3 + c];
            Complex rise = val_im1 - val_im2;
            solver_guess[p * 3 + c] = val_im1 + new_run * rise / run;
        }
    }

    // Predict quadrupoles
    if (solve_quadrupoles && num_quads > 0) {
        size_t solver_quad_offset = num_particles * 3;
        for (size_t q = 0; q < num_quads; ++q) {
            size_t p = quad_idxs.empty() ? q : quad_idxs[q];
            Complex eps_im2 = eps_p[im2][p];
            Complex eps_im1 = eps_p[im1][p];
            Complex eps_i = eps_p[i][p];

            Complex X_im2 = (eps_im2 - 1.0 == 0.0) ? 0.0 : (eps_im2 + 2.0) / (eps_im2 - 1.0);
            Complex X_im1 = (eps_im1 - 1.0 == 0.0) ? 0.0 : (eps_im1 + 2.0) / (eps_im1 - 1.0);
            Complex X_i = (eps_i - 1.0 == 0.0) ? 0.0 : (eps_i + 2.0) / (eps_i - 1.0);

            Complex run = X_im1 - X_im2;
            Complex new_run = X_i - X_im1;
            if (run == 0.0) {
                run = 1.0;
                new_run = 0.0;
            }
            for (int c = 0; c < 5; ++c) {
                Complex val_im1 = prev_quad[im1 * num_quads * 5 + q * 5 + c];
                Complex val_im2 = prev_quad[im2 * num_quads * 5 + q * 5 + c];
                Complex rise = val_im1 - val_im2;
                solver_guess[solver_quad_offset + q * 5 + c] = val_im1 + new_run * rise / run;
            }
        }
    }
    return solver_guess;
}

void EELS_Solver::compute(const std::vector<double>& epos,
                          const std::vector<double>& x_part,
                          const std::vector<double>& y_part,
                          const std::vector<double>& z_part) {
    if (x_part.size() != y_part.size() || x_part.size() != z_part.size()) {
        throw std::invalid_argument("Input coordinate vectors must have the exact same size.");
    }
    size_t num_p = x_part.size();
    if (num_p == 0) return;

    if (num_particles == 0) {
        set_dims(num_p);
        nondimensionalize();
        precalculations();
        if (!solver) {
            if (solver_type == "bicgstab") {
                solver = std::make_unique<BiCGSTAB_Solver>();
            } else {
                solver = std::make_unique<GMRES_Solver>();
            }
        }
        solver->initialize(num_particles * 3 + num_quads * 5);

        // Instantiate CUDA-accelerated Electric_Field solvers
        if (field_type == "direct") {
            EF = std::make_unique<Direct_Electric_Field>(radius, FieldCalcMode::SOLVER_AX, solve_quadrupoles, quad_idxs);
            eels_EF = std::make_unique<Direct_Electric_Field>(radius, FieldCalcMode::INTERACTION_FIELD, solve_quadrupoles, quad_idxs);
        } else {
            bool use_polydisperse = false;
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
                EF = std::make_unique<Polydisperse_Ewald_Electric_Field>(box[0], box[1], box[2], tol, xi, FieldCalcMode::SOLVER_AX, radius, solve_quadrupoles, quad_idxs);
                eels_EF = std::make_unique<Monodisperse_Ewald_Electric_Field>(box[0], box[1], box[2], tol, xi, FieldCalcMode::INTERACTION_FIELD, radius[0], solve_quadrupoles, quad_idxs);
            } else {
                EF = std::make_unique<Monodisperse_Ewald_Electric_Field>(box[0], box[1], box[2], tol, xi, FieldCalcMode::SOLVER_AX, radius[0], solve_quadrupoles, quad_idxs);
                eels_EF = std::make_unique<Monodisperse_Ewald_Electric_Field>(box[0], box[1], box[2], tol, xi, FieldCalcMode::INTERACTION_FIELD, radius[0], solve_quadrupoles, quad_idxs);
            }
        }
    } else if (num_particles != num_p) {
        throw std::runtime_error("The number of particles has changed!");
    }

    num_frames++;
    if (num_frames != 1) {
        print("frame");
        increase_indent();
    }

    std::vector<double> scaled_x(num_particles);
    std::vector<double> scaled_y(num_particles);
    std::vector<double> scaled_z(num_particles);
    for (size_t i = 0; i < num_particles; ++i) {
        scaled_x[i] = x_part[i] / length_scale;
        scaled_y[i] = y_part[i] / length_scale;
        scaled_z[i] = z_part[i] / length_scale;
    }

    if (num_frames != 1) {
        std::string frame_msg = std::to_string(num_frames - 1) + " of " + std::to_string(num_frames);
        print(frame_msg);
        increase_indent();
    }

    // Scale electron impact coordinates
    std::vector<double> scaled_epos(2);
    scaled_epos[0] = epos[0] / length_scale;
    scaled_epos[1] = epos[1] / length_scale;

    // Upload particle positions to both solvers
    EF->updateParticleCoordinates(scaled_x, scaled_y, scaled_z);
    eels_EF->updateParticleCoordinates(scaled_x, scaled_y, scaled_z);

    // Precompute EELS integration points and set target field points
    precomp_eels(scaled_z);
    size_t num_field_points = z_pts.size();
    std::vector<double> field_x(num_field_points, scaled_epos[0]);
    std::vector<double> field_y(num_field_points, scaled_epos[1]);
    std::vector<double> field_z = z_pts;

    if (field_type == "direct") {
        static_cast<Direct_Electric_Field*>(eels_EF.get())->updateFieldCoordinates(field_x, field_y, field_z);
    } else if (dynamic_cast<Polydisperse_Ewald_Electric_Field*>(eels_EF.get()) != nullptr) {
        static_cast<Polydisperse_Ewald_Electric_Field*>(eels_EF.get())->updateFieldCoordinates(field_x, field_y, field_z);
    } else {
        // Ewald
        static_cast<Monodisperse_Ewald_Electric_Field*>(eels_EF.get())->updateFieldCoordinates(field_x, field_y, field_z);
    }

    // Allocate results for this frame
    std::vector<Complex> frame_dip(num_wavevectors * num_particles * 3, 0.0);
    std::vector<Complex> frame_quad(num_wavevectors * num_quads * 5, 0.0);
    std::vector<double> frame_eels(num_wavevectors, 0.0);
    std::vector<Complex> dip_sum(num_wavevectors * num_particles * 3, 0.0);
    std::vector<Complex> quad_sum(num_wavevectors * num_quads * 5, 0.0);

    if (asm_flag) {
        print("Brillouin Zone point:");
    } else {
        print("Wavenumber:");
    }
    increase_indent();

    Complex gamma = 1.0 / std::sqrt(Complex(1.0 - eps_m * v * v));
    double min_frac = 0.2;

    std::vector<double> eels_sum(num_wavevectors, 0.0);

    int Nx = asm_flag ? asm_Nx : 1;
    int Ny = asm_flag ? asm_Ny : 1;
    double dkx = 0.0;
    double dky = 0.0;
    if (asm_flag) {
        dkx = 2.0 * M_PI / (box[0] * Nx);
        dky = 2.0 * M_PI / (box[1] * Ny);
    }

    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            double k_x = 0.0;
            double k_y = 0.0;
            if (asm_flag) {
                k_x = -M_PI / box[0] + (ix + 0.5) * dkx;
                k_y = -M_PI / box[1] + (iy + 0.5) * dky;
            }

            if (auto* ewald_ef = dynamic_cast<Ewald_Electric_Field_Base*>(EF.get())) {
                ewald_ef->setBlochWavevector(k_x, k_y);
            }
            if (auto* ewald_eels_ef = dynamic_cast<Ewald_Electric_Field_Base*>(eels_EF.get())) {
                ewald_eels_ef->setBlochWavevector(k_x, k_y);
            }

            for (size_t wavevec_idx = 0; wavevec_idx < num_wavevectors; ++wavevec_idx) {
        if (asm_flag) {
            if (wavevec_idx == 0) {
                std::string progress = std::to_string(ix * Ny + iy) + " of " + std::to_string(Nx * Ny);
                print(progress);
            }
        } else {
            std::string progress = std::to_string(wavevec_idx) + " of " + std::to_string(num_wavevectors);
            print(progress);
        }

        double omega_val = omega[wavevec_idx];

        // 1. Calculate space-frequency relativistic ebeam field at each particle
        int R_max_x = 0;
        int R_max_y = 0;
        if (asm_flag) {
            double r_decay = 12.0 * std::abs((v * gamma) / (2.0 * M_PI * omega_val));
            R_max_x = static_cast<int>(std::ceil(r_decay / box[0]));
            R_max_y = static_cast<int>(std::ceil(r_decay / box[1]));
        }

        std::vector<Complex> E_inc(num_particles * 3 + num_quads * 5, 0.0);
        for (int ix_R = -R_max_x; ix_R <= R_max_x; ++ix_R) {
            for (int iy_R = -R_max_y; iy_R <= R_max_y; ++iy_R) {
                double Rx = ix_R * box[0];
                double Ry = iy_R * box[1];

                for (size_t p = 0; p < num_particles; ++p) {
                    double dx = scaled_x[p] - scaled_epos[0] - Rx;
                    double dy = scaled_y[p] - scaled_epos[1] - Ry;
                    double r = std::sqrt(dx*dx + dy*dy);
                    if (r < min_frac) r = min_frac;
                    double rhat_x = dx / r;
                    double rhat_y = dy / r;

                    Complex xi_val = (2.0 * M_PI * omega_val * r) / (v * gamma);
                    Complex prefactor = (4.0 * M_PI * omega_val) / (v * v * gamma) * std::exp(Complex(0.0, 2.0 * M_PI * omega_val * scaled_z[p] / v));
                    
                    Complex k0 = eval_k0(xi_val);
                    Complex k1 = eval_k1(xi_val);

                    Complex E_x = -prefactor * k1 * rhat_x;
                    Complex E_y = -prefactor * k1 * rhat_y;
                    Complex E_z = prefactor * Complex(0.0, 1.0) / gamma * k0;

                    Complex phase_factor(1.0, 0.0);
                    if (asm_flag) {
                        double phase = k_x * (scaled_x[p] - Rx) + k_y * (scaled_y[p] - Ry);
                        phase_factor = std::exp(Complex(0.0, phase));
                    }

                    E_inc[p * 3 + 0] += E_x * phase_factor;
                    E_inc[p * 3 + 1] += E_y * phase_factor;
                    E_inc[p * 3 + 2] += E_z * phase_factor;
                }

                if (solve_quadrupoles && num_quads > 0) {
                    for (size_t q = 0; q < num_quads; ++q) {
                        size_t p = quad_idxs.empty() ? q : quad_idxs[q];
                        double dx = scaled_x[p] - scaled_epos[0] - Rx;
                        double dy = scaled_y[p] - scaled_epos[1] - Ry;
                        double r = std::sqrt(dx*dx + dy*dy);
                        if (r < min_frac) r = min_frac;
                        double rhat_x = dx / r;
                        double rhat_y = dy / r;

                        Complex xi_val = (2.0 * M_PI * omega_val * r) / (v * gamma);
                        Complex beta = (2.0 * M_PI * omega_val) / (v * gamma);
                        Complex prefactor = (4.0 * M_PI * omega_val) / (v * v * gamma) * std::exp(Complex(0.0, 2.0 * M_PI * omega_val * scaled_z[p] / v));
                        
                        Complex k0 = eval_k0(xi_val);
                        Complex k1 = eval_k1(xi_val);

                        Complex G_xx = prefactor * (beta * k0 * rhat_x * rhat_x - ((1.0 - 2.0 * rhat_x * rhat_x) / r) * k1);
                        Complex G_yy = prefactor * (beta * k0 * rhat_y * rhat_y - ((1.0 - 2.0 * rhat_y * rhat_y) / r) * k1);
                        Complex G_zz = -beta * prefactor * k0;
                        Complex G_xy = prefactor * (beta * k0 * rhat_x * rhat_y + (2.0 * rhat_x * rhat_y / r) * k1);
                        Complex G_xz = -Complex(0.0, 1.0) * 0.5 * prefactor * beta * (gamma + 1.0 / gamma) * k1 * rhat_x;
                        Complex G_yz = -Complex(0.0, 1.0) * 0.5 * prefactor * beta * (gamma + 1.0 / gamma) * k1 * rhat_y;

                        Complex phase_factor(1.0, 0.0);
                        if (asm_flag) {
                            double phase = k_x * (scaled_x[p] - Rx) + k_y * (scaled_y[p] - Ry);
                            phase_factor = std::exp(Complex(0.0, phase));
                        }

                        E_inc[num_particles * 3 + q * 5 + 0] += (G_xx - G_zz) * phase_factor;
                        E_inc[num_particles * 3 + q * 5 + 1] += G_xy * phase_factor;
                        E_inc[num_particles * 3 + q * 5 + 2] += G_xz * phase_factor;
                        E_inc[num_particles * 3 + q * 5 + 3] += (G_yy - G_zz) * phase_factor;
                        E_inc[num_particles * 3 + q * 5 + 4] += G_yz * phase_factor;
                    }
                }
            }
        }

        // 2. Set self coefs on EF
        std::vector<double> self_coef_r(num_particles);
        std::vector<double> self_coef_i(num_particles);
        for (size_t p = 0; p < num_particles; ++p) {
            Complex sc = -3.0 / (4.0 * M_PI * (1.0 - eps_p[wavevec_idx][p]));
            self_coef_r[p] = sc.real();
            self_coef_i[p] = sc.imag();
        }
        EF->setSelfCoef(self_coef_r, self_coef_i);

        // 3. Normalize incident field for numerical stability
        double Enorm = 0.0;
        for (size_t p = 0; p < num_particles; ++p) {
            double norm = std::sqrt(std::norm(E_inc[p * 3 + 0]) + std::norm(E_inc[p * 3 + 1]) + std::norm(E_inc[p * 3 + 2]));
            if (norm > Enorm) Enorm = norm;
        }
        if (solve_quadrupoles && num_quads > 0) {
            for (size_t q = 0; q < num_quads; ++q) {
                double norm = std::sqrt(std::norm(E_inc[num_particles * 3 + q * 5 + 0]) + 
                                        std::norm(E_inc[num_particles * 3 + q * 5 + 1]) + 
                                        std::norm(E_inc[num_particles * 3 + q * 5 + 2]) + 
                                        std::norm(E_inc[num_particles * 3 + q * 5 + 3]) + 
                                        std::norm(E_inc[num_particles * 3 + q * 5 + 4]));
                if (norm > Enorm) Enorm = norm;
            }
        }
        if (Enorm < 1e-15) Enorm = 1.0;

        std::vector<Complex> E_inc_norm(num_particles * 3 + num_quads * 5);
        for (size_t i = 0; i < num_particles * 3 + num_quads * 5; ++i) {
            E_inc_norm[i] = E_inc[i] / Enorm;
        }

        // 4. Calculate initial guess and solve system A * p = E
        std::vector<Complex> solver_guess;
        if (wavevec_idx == 0) {
            // Match python's first wavelength guess alignment step:
            // Ew = Ew / norm(Ew), solver_guess = 0.1 * Ew * sign(Ew), dips[:,2] *= 1j
            // Scaled by mean field guess: 4 * pi * beta / (1 - beta * vol_frac)
            double sum_r3 = 0.0;
            for (double r : radius) {
                sum_r3 += std::pow(r, 3.0);
            }
            double box_vol = box[0] * box[1] * box[2];
            double vol_frac = (4.0 / 3.0) * M_PI * sum_r3 / box_vol;

            std::vector<Complex> Ew_norm(num_particles * 3);
            for (size_t p = 0; p < num_particles; ++p) {
                double norm_p = std::sqrt(std::norm(E_inc[p * 3 + 0]) + std::norm(E_inc[p * 3 + 1]) + std::norm(E_inc[p * 3 + 2]));
                if (norm_p < 1e-15) norm_p = 1.0;
                Ew_norm[p * 3 + 0] = E_inc[p * 3 + 0] / norm_p;
                Ew_norm[p * 3 + 1] = E_inc[p * 3 + 1] / norm_p;
                Ew_norm[p * 3 + 2] = E_inc[p * 3 + 2] / norm_p;
            }
            solver_guess.resize(num_particles * 3 + num_quads * 5, 0.0);
            for (size_t p = 0; p < num_particles; ++p) {
                Complex beta = (eps_p[0][p] - 1.0) / (eps_p[0][p] + 2.0);
                Complex val = 4.0 * M_PI * std::pow(radius[p], 3.0) * beta / (1.0 - beta * vol_frac);
                for (int c = 0; c < 3; ++c) {
                    Complex ew = Ew_norm[p * 3 + c];
                    double sgn = (ew.real() >= 0.0) ? 1.0 : -1.0;
                    solver_guess[p * 3 + c] = 0.1 * val * ew * sgn;
                }
                solver_guess[p * 3 + 2] *= Complex(0.0, 1.0);
            }
        } else {
            solver_guess = calc_guess(frame_dip, frame_quad, wavevec_idx, E_inc_norm);
        }

        // Divide initial guess by Enorm as well
        for (size_t i = 0; i < num_particles * 3 + num_quads * 5; ++i) {
            solver_guess[i] /= Enorm;
        }

        std::vector<Complex> sol_norm = solver->solve(E_inc_norm, solver_guess, EF.get(), tol, quiet);

        // Scale solved dipoles and quadrupoles back by Enorm and save
        for (size_t i = 0; i < num_particles * 3; ++i) {
            frame_dip[wavevec_idx * num_particles * 3 + i] = sol_norm[i] * Enorm;
            dip_sum[wavevec_idx * num_particles * 3 + i] += sol_norm[i] * Enorm;
        }
        if (solve_quadrupoles && num_quads > 0) {
            for (size_t i = 0; i < num_quads * 5; ++i) {
                frame_quad[wavevec_idx * num_quads * 5 + i] = sol_norm[num_particles * 3 + i] * Enorm;
                quad_sum[wavevec_idx * num_quads * 5 + i] += sol_norm[num_particles * 3 + i] * Enorm;
            }
        }

        // 5. Evaluate EELS probability using eels_EF
        std::vector<double> dips_xr(num_particles), dips_xi(num_particles);
        std::vector<double> dips_yr(num_particles), dips_yi(num_particles);
        std::vector<double> dips_zr(num_particles), dips_zi(num_particles);
        for (size_t p = 0; p < num_particles; ++p) {
            Complex d_val = sol_norm[p * 3 + 0] * Enorm;
            dips_xr[p] = d_val.real();
            dips_xi[p] = d_val.imag();
            d_val = sol_norm[p * 3 + 1] * Enorm;
            dips_yr[p] = d_val.real();
            dips_yi[p] = d_val.imag();
            d_val = sol_norm[p * 3 + 2] * Enorm;
            dips_zr[p] = d_val.real();
            dips_zi[p] = d_val.imag();
        }

        if (field_type == "direct") {
            auto* derived = static_cast<Direct_Electric_Field*>(eels_EF.get());
            derived->updateDipolesComplex(dips_xr, dips_xi, dips_yr, dips_yi, dips_zr, dips_zi);
            if (solve_quadrupoles && num_quads > 0) {
                std::vector<double> q1r(num_quads), q1i(num_quads), q2r(num_quads), q2i(num_quads), q3r(num_quads), q3i(num_quads), q4r(num_quads), q4i(num_quads), q5r(num_quads), q5i(num_quads);
                for (size_t q = 0; q < num_quads; ++q) {
                    Complex q_val = sol_norm[num_particles * 3 + q * 5 + 0] * Enorm;
                    q1r[q] = q_val.real(); q1i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 1] * Enorm;
                    q2r[q] = q_val.real(); q2i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 2] * Enorm;
                    q3r[q] = q_val.real(); q3i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 3] * Enorm;
                    q4r[q] = q_val.real(); q4i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 4] * Enorm;
                    q5r[q] = q_val.real(); q5i[q] = q_val.imag();
                }
                derived->updateQuadrupolesComplex(q1r, q1i, q2r, q2i, q3r, q3i, q4r, q4i, q5r, q5i);
            }
            std::vector<double> self_r(num_particles, 0.0), self_i(num_particles, 0.0);
            derived->setSelfCoef(self_r, self_i);
        } else if (dynamic_cast<Polydisperse_Ewald_Electric_Field*>(eels_EF.get()) != nullptr) {
            auto* derived = static_cast<Polydisperse_Ewald_Electric_Field*>(eels_EF.get());
            derived->updateDipolesComplex(dips_xr, dips_xi, dips_yr, dips_yi, dips_zr, dips_zi);
            if (solve_quadrupoles && num_quads > 0) {
                std::vector<double> q1r(num_quads), q1i(num_quads), q2r(num_quads), q2i(num_quads), q3r(num_quads), q3i(num_quads), q4r(num_quads), q4i(num_quads), q5r(num_quads), q5i(num_quads);
                for (size_t q = 0; q < num_quads; ++q) {
                    Complex q_val = sol_norm[num_particles * 3 + q * 5 + 0] * Enorm;
                    q1r[q] = q_val.real(); q1i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 1] * Enorm;
                    q2r[q] = q_val.real(); q2i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 2] * Enorm;
                    q3r[q] = q_val.real(); q3i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 3] * Enorm;
                    q4r[q] = q_val.real(); q4i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 4] * Enorm;
                    q5r[q] = q_val.real(); q5i[q] = q_val.imag();
                }
                derived->updateQuadrupolesComplex(q1r, q1i, q2r, q2i, q3r, q3i, q4r, q4i, q5r, q5i);
            }
            std::vector<double> self_r(num_particles, 0.0), self_i(num_particles, 0.0);
            derived->setSelfCoef(self_r, self_i);
        } else {
            auto* derived = static_cast<Monodisperse_Ewald_Electric_Field*>(eels_EF.get());
            derived->updateDipolesComplex(dips_xr, dips_xi, dips_yr, dips_yi, dips_zr, dips_zi);
            if (solve_quadrupoles && num_quads > 0) {
                std::vector<double> q1r(num_quads), q1i(num_quads), q2r(num_quads), q2i(num_quads), q3r(num_quads), q3i(num_quads), q4r(num_quads), q4i(num_quads), q5r(num_quads), q5i(num_quads);
                for (size_t q = 0; q < num_quads; ++q) {
                    Complex q_val = sol_norm[num_particles * 3 + q * 5 + 0] * Enorm;
                    q1r[q] = q_val.real(); q1i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 1] * Enorm;
                    q2r[q] = q_val.real(); q2i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 2] * Enorm;
                    q3r[q] = q_val.real(); q3i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 3] * Enorm;
                    q4r[q] = q_val.real(); q4i[q] = q_val.imag();
                    q_val = sol_norm[num_particles * 3 + q * 5 + 4] * Enorm;
                    q5r[q] = q_val.real(); q5i[q] = q_val.imag();
                }
                derived->updateQuadrupolesComplex(q1r, q1i, q2r, q2i, q3r, q3i, q4r, q4i, q5r, q5i);
            }
            std::vector<double> self_r(num_particles, 0.0), self_i(num_particles, 0.0);
            derived->setSelfCoef(self_r, self_i);
        }

        eels_EF->calculate();

        // 6. Copy evaluated field back to host
        std::vector<double> host_E_point(num_field_points * 3 * 2);
        CUDA_CHECK(cudaMemcpy(host_E_point.data(), eels_EF->getDevEPoint(), num_field_points * 3 * 2 * sizeof(double), cudaMemcpyDeviceToHost));

        // 7. Simpson's integration
        std::vector<double> integrand(num_field_points);
        for (size_t j = 0; j < num_field_points; ++j) {
            double Ez_r = host_E_point[(j * 3 + 2) * 2 + 0];
            double Ez_i = host_E_point[(j * 3 + 2) * 2 + 1];
            Complex Eind_z = Complex(Ez_r, Ez_i);

            double phase = -2.0 * M_PI * omega_val * Z_pts[j] / v;
            double k_phase = -(k_x * scaled_epos[0] + k_y * scaled_epos[1]);
            Complex exp_factor = std::exp(Complex(0.0, phase + k_phase));
            integrand[j] = (Eind_z * exp_factor).real();
        }

        double integral = simpson_integrate(integrand, integration_step);
        double eels_val = integral / (2.0 * M_PI * M_PI * omega_val);
        
        // The solved dipoles were already physically scaled by Enorm,
        // so eels_val already contains the Enorm scaling factor.
        frame_eels[wavevec_idx] = eels_val;
        eels_sum[wavevec_idx] += eels_val;
    }
    }
    }
    
    if (asm_flag) {
        double dk_area = dkx * dky;
        double bz_area = (2.0 * M_PI / box[0]) * (2.0 * M_PI / box[1]);
        for (size_t wavevec_idx = 0; wavevec_idx < num_wavevectors; ++wavevec_idx) {
            frame_eels[wavevec_idx] = eels_sum[wavevec_idx] * dk_area / bz_area;
            for (size_t i = 0; i < num_particles * 3; ++i) {
                dip_sum[wavevec_idx * num_particles * 3 + i] *= (dk_area / bz_area);
            }
            if (solve_quadrupoles && num_quads > 0) {
                for (size_t i = 0; i < num_quads * 5; ++i) {
                    quad_sum[wavevec_idx * num_quads * 5 + i] *= (dk_area / bz_area);
                }
            }
        }
    }
    decrease_indent();

    eels.insert(eels.end(), frame_eels.begin(), frame_eels.end());
    if (asm_flag) {
        dips.insert(dips.end(), dip_sum.begin(), dip_sum.end());
        if (solve_quadrupoles && num_quads > 0) {
            quads.insert(quads.end(), quad_sum.begin(), quad_sum.end());
        }
    } else {
        dips.insert(dips.end(), frame_dip.begin(), frame_dip.end());
        if (solve_quadrupoles && num_quads > 0) {
            quads.insert(quads.end(), frame_quad.begin(), frame_quad.end());
        }
    }

    if (num_frames != 1) {
        decrease_indent();
        decrease_indent();
    }
}
