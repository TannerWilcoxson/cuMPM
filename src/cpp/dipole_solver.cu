#include "dipole_solver.h"
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
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA Error Checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::string err_msg = "CUDA Error: " + std::string(cudaGetErrorString(code)) + " at " + std::string(file) + ":" + std::to_string(line);
      if (abort) {
         throw std::runtime_error(err_msg);
      } else {
         std::cerr << err_msg << "\n";
      }
   }
}

// -----------------------------------------------------------------------------
// Dipole_Solver Class Implementation
// -----------------------------------------------------------------------------

// Constructor 1: Full 2D eps_p
Dipole_Solver::Dipole_Solver(const std::vector<double>& box,
                             const std::vector<std::vector<Complex>>& eps_p,
                             const std::vector<double>& radius,
                             double eps_m,
                             double xi,
                             double tol,
                             bool quiet,
                             const std::string& guess_type,
                             const std::string& solver_type,
                             const std::string& field_type,
                             const std::vector<std::vector<Complex>>& E0,
                             bool solve_quadrupoles,
                             const std::vector<int>& quad_idxs)
    : box(box), eps_p(eps_p), radius(radius), eps_m(eps_m), xi(xi), tol(tol), quiet(quiet), guess_type(guess_type), solver_type(solver_type), field_type(field_type), E0(E0), solve_quadrupoles(solve_quadrupoles), quad_idxs(quad_idxs) {
    if (this->E0.empty()) {
        this->E0 = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
    }
    // E0 validation will be performed dynamically in set_dims() once N and Q are known
}

// Constructor 2: Wavelength-dependent (1D eps_p)
Dipole_Solver::Dipole_Solver(const std::vector<double>& box,
                             const std::vector<Complex>& eps_p_1d,
                             const std::vector<double>& radius,
                             double eps_m,
                             double xi,
                             double tol,
                             bool quiet,
                             const std::string& guess_type,
                             const std::string& solver_type,
                             const std::string& field_type,
                             const std::vector<std::vector<Complex>>& E0,
                             bool solve_quadrupoles,
                             const std::vector<int>& quad_idxs)
    : box(box), radius(radius), eps_m(eps_m), xi(xi), tol(tol), quiet(quiet), guess_type(guess_type), solver_type(solver_type), field_type(field_type), E0(E0), solve_quadrupoles(solve_quadrupoles), quad_idxs(quad_idxs) {
    if (this->E0.empty()) {
        this->E0 = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
    }
    // E0 validation will be performed dynamically in set_dims() once N and Q are known
    
    // We will expand eps_p_1d in compute() once we know num_particles
    this->eps_p.resize(eps_p_1d.size(), std::vector<Complex>(1, 0.0));
    for (size_t w = 0; w < eps_p_1d.size(); ++w) {
        this->eps_p[w][0] = eps_p_1d[w];
    }
}

// Constructor 3: Scalar eps_p
Dipole_Solver::Dipole_Solver(const std::vector<double>& box,
                             Complex eps_p_scalar,
                             const std::vector<double>& radius,
                             double eps_m,
                             double xi,
                             double tol,
                             bool quiet,
                             const std::string& guess_type,
                             const std::string& solver_type,
                             const std::string& field_type,
                             const std::vector<std::vector<Complex>>& E0,
                             bool solve_quadrupoles,
                             const std::vector<int>& quad_idxs)
    : box(box), radius(radius), eps_m(eps_m), xi(xi), tol(tol), quiet(quiet), guess_type(guess_type), solver_type(solver_type), field_type(field_type), E0(E0), solve_quadrupoles(solve_quadrupoles), quad_idxs(quad_idxs) {
    if (this->E0.empty()) {
        this->E0 = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
    }
    // E0 validation will be performed dynamically in set_dims() once N and Q are known
    
    this->eps_p.resize(1, std::vector<Complex>(1, eps_p_scalar));
}

Dipole_Solver::~Dipole_Solver() {
}

void Dipole_Solver::print(const std::string& msg) const {
    if (!quiet) {
        std::string indent(indent_level * 2, ' ');
        std::cout << indent << msg << std::endl;
    }
}

void Dipole_Solver::set_dims(size_t num_p) {
    num_particles = num_p;
    num_quads = solve_quadrupoles ? (quad_idxs.empty() ? num_particles : quad_idxs.size()) : 0;

    // Set default radius if empty or scalar
    if (radius.empty()) {
        radius.resize(num_particles, 1.0);
    } else if (radius.size() == 1) {
        double val = radius[0];
        radius.resize(num_particles, val);
    } else if (radius.size() != num_particles) {
        throw std::runtime_error("The number of particles is inconsistent with the number of radii provided.");
    }

    // Check if radii are identical if monodisperse is requested
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

    // Expand eps_p if it was 1D or scalar
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

    // Dynamically validate custom incident field (E0) dimensions
    size_t vec_size = num_particles * 3 + num_quads * 5;
    for (size_t dim = 0; dim < E0.size(); ++dim) {
        if (E0[dim].size() != 3 && E0[dim].size() != num_particles * 3 && E0[dim].size() != vec_size) {
            throw std::runtime_error("Incident field E0[" + std::to_string(dim) + "] has size " + 
                                     std::to_string(E0[dim].size()) + ", but expected 3, " + 
                                     std::to_string(num_particles * 3) + ", or " + 
                                     std::to_string(vec_size) + ".");
        }
    }
}

void Dipole_Solver::nondimensionalize() {
    length_scale = radius[0];
    eps_scale = eps_m;

    if (box.size() >= 3) {
        box[0] /= length_scale;
        box[1] /= length_scale;
        box[2] /= length_scale;
    }

    for (size_t i = 0; i < radius.size(); ++i) {
        radius[i] /= length_scale;
    }

    eps_m /= eps_scale; // becomes 1.0

    for (size_t w = 0; w < num_wavevectors; ++w) {
        for (size_t p = 0; p < num_particles; ++p) {
            eps_p[w][p] /= eps_scale;
        }
    }
}

void Dipole_Solver::calc_vol_frac() {
    double sum_r3 = 0.0;
    for (double r : radius) {
        sum_r3 += std::pow(r, 3.0);
    }
    double box_vol = 0.0;
    if (box.size() >= 3) {
        box_vol = box[0] * box[1] * box[2];
    }
    if (box_vol == 0.0) {
        vol_frac = 0.0;
    } else {
        vol_frac = (4.0 / 3.0) * M_PI * sum_r3 / box_vol;
    }
}

void Dipole_Solver::precalculations() {
    calc_vol_frac();
}



// -----------------------------------------------------------------------------
// Guess Predictor Calculations
// -----------------------------------------------------------------------------

std::vector<Complex> Dipole_Solver::calc_guess(const std::vector<Complex>& prev_dip, const std::vector<Complex>& prev_quad, size_t wavevec_idx) const {
    if (guess_type == "mean_field" || guess_type == "mean-field") {
        return calc_mean_field_guess(wavevec_idx);
    } else if (guess_type == "previous") {
        return calc_previous_guess(prev_dip, prev_quad, wavevec_idx);
    } else if (guess_type == "derivative") {
        return calc_derivative_guess(prev_dip, prev_quad, wavevec_idx);
    } else {
        throw std::runtime_error("Guess type " + guess_type + " not supported.");
    }
}

std::vector<Complex> Dipole_Solver::calc_mean_field_guess(size_t wavevec_idx) const {
    size_t K = E0.size();
    size_t total_size = num_particles * K * 3 + num_quads * K * 5;
    std::vector<Complex> dip_guess(total_size, 0.0);
    for (size_t p = 0; p < num_particles; ++p) {
        Complex beta = (eps_p[wavevec_idx][p] - 1.0) / (eps_p[wavevec_idx][p] + 2.0);
        Complex val = 4.0 * M_PI * std::pow(radius[p], 3.0) * beta / (1.0 - beta * vol_frac);
        
        for (size_t dim = 0; dim < K; ++dim) {
            for (int c = 0; c < 3; ++c) {
                Complex e_inc = (E0[dim].size() > 3) ? E0[dim][p * 3 + c] : E0[dim][c];
                dip_guess[p * (K * 3) + dim * 3 + c] = val * e_inc;
            }
        }
    }
    return dip_guess;
}

std::vector<Complex> Dipole_Solver::calc_previous_guess(const std::vector<Complex>& prev_dip, const std::vector<Complex>& prev_quad, size_t wavevec_idx) const {
    if (wavevec_idx < 1) {
        return calc_mean_field_guess(wavevec_idx);
    }
    size_t K = E0.size();
    size_t total_size = num_particles * K * 3 + num_quads * K * 5;
    std::vector<Complex> dip_guess(total_size);
    size_t offset = (wavevec_idx - 1) * num_particles * K * 3;
    std::copy(prev_dip.begin() + offset, prev_dip.begin() + offset + num_particles * K * 3, dip_guess.begin());
    if (solve_quadrupoles && num_quads > 0) {
        size_t quad_offset = (wavevec_idx - 1) * num_quads * K * 5;
        std::copy(prev_quad.begin() + quad_offset, prev_quad.begin() + quad_offset + num_quads * K * 5, dip_guess.begin() + num_particles * K * 3);
    }
    return dip_guess;
}

std::vector<Complex> Dipole_Solver::calc_derivative_guess(const std::vector<Complex>& prev_dip, const std::vector<Complex>& prev_quad, size_t wavevec_idx) const {
    if (wavevec_idx < 1) {
        return calc_mean_field_guess(wavevec_idx);
    }
    if (wavevec_idx < 2) {
        return calc_previous_guess(prev_dip, prev_quad, wavevec_idx);
    }
    size_t K = E0.size();
    size_t i = wavevec_idx;
    size_t im2 = i - 2;
    size_t im1 = i - 1;

    size_t total_size = num_particles * K * 3 + num_quads * K * 5;
    std::vector<Complex> dip_guess(total_size);
    for (size_t p = 0; p < num_particles; ++p) {
        Complex eps_im2 = eps_p[im2][p];
        Complex eps_im1 = eps_p[im1][p];
        Complex eps_i   = eps_p[i][p];

        // Use inverse polarizability X = (eps+2)/(eps-1) as the extrapolation variable
        Complex X_im2 = (eps_im2 - 1.0 == 0.0) ? 0.0 : (eps_im2 + 2.0) / (eps_im2 - 1.0);
        Complex X_im1 = (eps_im1 - 1.0 == 0.0) ? 0.0 : (eps_im1 + 2.0) / (eps_im1 - 1.0);
        Complex X_i   = (eps_i   - 1.0 == 0.0) ? 0.0 : (eps_i   + 2.0) / (eps_i   - 1.0);

        Complex run = X_im1 - X_im2;
        Complex new_run = X_i - X_im1;
        if (std::abs(run) < 1e-12) {
            run = 1.0;
            new_run = 0.0;
        }

        for (size_t idx = 0; idx < K * 3; ++idx) {
            Complex val_im1 = prev_dip[im1 * num_particles * K * 3 + p * K * 3 + idx];
            Complex val_im2 = prev_dip[im2 * num_particles * K * 3 + p * K * 3 + idx];
            Complex rise = val_im1 - val_im2;
            dip_guess[p * K * 3 + idx] = val_im1 + new_run * rise / run;
        }
    }

    if (solve_quadrupoles && num_quads > 0) {
        size_t quad_offset = num_particles * K * 3;
        for (size_t q = 0; q < num_quads; ++q) {
            size_t p = quad_idxs.empty() ? q : quad_idxs[q];
            Complex eps_im2 = eps_p[im2][p];
            Complex eps_im1 = eps_p[im1][p];
            Complex eps_i   = eps_p[i][p];

            Complex X_im2 = (eps_im2 - 1.0 == 0.0) ? 0.0 : (eps_im2 + 2.0) / (eps_im2 - 1.0);
            Complex X_im1 = (eps_im1 - 1.0 == 0.0) ? 0.0 : (eps_im1 + 2.0) / (eps_im1 - 1.0);
            Complex X_i   = (eps_i   - 1.0 == 0.0) ? 0.0 : (eps_i   + 2.0) / (eps_i   - 1.0);

            Complex run = X_im1 - X_im2;
            Complex new_run = X_i - X_im1;
            if (std::abs(run) < 1e-12) {
                run = 1.0;
                new_run = 0.0;
            }

            for (size_t idx = 0; idx < K * 5; ++idx) {
                Complex val_im1 = prev_quad[im1 * num_quads * K * 5 + q * K * 5 + idx];
                Complex val_im2 = prev_quad[im2 * num_quads * K * 5 + q * K * 5 + idx];
                Complex rise = val_im1 - val_im2;
                dip_guess[quad_offset + q * K * 5 + idx] = val_im1 + new_run * rise / run;
            }
        }
    }
    return dip_guess;
}

// -----------------------------------------------------------------------------
// GMRES Solver Core (GPU-Resident)
// -----------------------------------------------------------------------------

std::vector<Complex> Dipole_Solver::compute_dipoles(const std::vector<Complex>& E, 
                                                    const std::vector<Complex>& dip_guess) {
    return solver->solve(E, dip_guess, EF.get(), tol);
}

// -----------------------------------------------------------------------------
// Spectrum and Tensor computation loops
// -----------------------------------------------------------------------------

void Dipole_Solver::compute_tensor(const std::vector<Complex>& dip_guess, 
                                   std::vector<Complex>& frame_cap, 
                                   std::vector<Complex>& frame_dip,
                                   std::vector<Complex>& frame_quad) {
    size_t vec_size = num_particles * 3 + num_quads * 5;
    size_t K = E0.size();
    
    // E represents unit fields or custom fields for K polarization directions
    std::vector<std::vector<Complex>> E(K, std::vector<Complex>(vec_size, 0.0));
    for (size_t dim = 0; dim < K; ++dim) {
        if (E0[dim].size() == num_particles * 3) {
            for (size_t i = 0; i < num_particles * 3; ++i) {
                E[dim][i] = E0[dim][i];
            }
        } else if (E0[dim].size() == 3) {
            for (size_t p = 0; p < num_particles; ++p) {
                E[dim][p * 3 + 0] = E0[dim][0];
                E[dim][p * 3 + 1] = E0[dim][1];
                E[dim][p * 3 + 2] = E0[dim][2];
            }
        } else if (E0[dim].size() == vec_size) {
            E[dim] = E0[dim];
        } else {
            throw std::runtime_error("Incident field E0[" + std::to_string(dim) + "] has size " + 
                                     std::to_string(E0[dim].size()) + ", but expected either 3 or " + 
                                     std::to_string(num_particles * 3) + ".");
        }
    }

    // Solve for each orientation
    for (size_t dim = 0; dim < K; ++dim) {
        std::vector<Complex> dip_guess_dim(vec_size, 0.0);
        for (size_t p = 0; p < num_particles; ++p) {
            for (int c = 0; c < 3; ++c) {
                dip_guess_dim[p * 3 + c] = dip_guess[p * (K * 3) + dim * 3 + c];
            }
        }
        if (solve_quadrupoles && num_quads > 0) {
            size_t quad_offset = num_particles * K * 3;
            for (size_t q = 0; q < num_quads; ++q) {
                for (int c = 0; c < 5; ++c) {
                    dip_guess_dim[num_particles * 3 + q * 5 + c] = dip_guess[quad_offset + q * (K * 5) + dim * 5 + c];
                }
            }
        }

        std::vector<Complex> sol = compute_dipoles(E[dim], dip_guess_dim);

        // Store solved dipoles
        for (size_t p = 0; p < num_particles; ++p) {
            for (int c = 0; c < 3; ++c) {
                frame_dip[p * (K * 3) + dim * 3 + c] = sol[p * 3 + c];
            }
        }

        // Store solved quadrupoles
        if (solve_quadrupoles && num_quads > 0) {
            for (size_t q = 0; q < num_quads; ++q) {
                for (int c = 0; c < 5; ++c) {
                    frame_quad[q * (K * 5) + dim * 5 + c] = sol[num_particles * 3 + q * 5 + c];
                }
            }
        }

        // Average dipoles to get cap (tensor dimension dim)
        std::vector<Complex> avg_dip_dim(3, 0.0);
        for (size_t p = 0; p < num_particles; ++p) {
            avg_dip_dim[0] += sol[p * 3 + 0];
            avg_dip_dim[1] += sol[p * 3 + 1];
            avg_dip_dim[2] += sol[p * 3 + 2];
        }
        for (int c = 0; c < 3; ++c) {
            frame_cap[dim * 3 + c] = avg_dip_dim[c] / static_cast<double>(num_particles);
        }
    }
}

std::vector<Complex> Dipole_Solver::compute_spectrum(const std::vector<Complex>& initial_guess, std::vector<Complex>& frame_quad) {
    size_t K = E0.size();
    std::vector<Complex> frame_cap(num_wavevectors * K * 3, 0.0);
    std::vector<Complex> frame_dip(num_wavevectors * num_particles * K * 3, 0.0);
    frame_quad.assign(num_wavevectors * num_quads * K * 5, 0.0);

    std::vector<Complex> dip_guess = initial_guess;

    print("Wavenumber:");
    increase_indent();
    for (size_t wavevec_idx = 0; wavevec_idx < num_wavevectors; ++wavevec_idx) {
        std::string progress = std::to_string(wavevec_idx) + " of " + std::to_string(num_wavevectors);
        print(progress);

        // Calculate complex particle-dependent Ewald self coefficients
        std::vector<double> self_coef_r(num_particles);
        std::vector<double> self_coef_i(num_particles);
        for (size_t p = 0; p < num_particles; ++p) {
            Complex diff = 1.0 - eps_p[wavevec_idx][p];
            if (std::abs(diff) < 1e-12) {
                diff = 1e-12;
            }
            Complex sc = -3.0 / (4.0 * M_PI * diff) / std::pow(radius[p], 3.0);
            self_coef_r[p] = sc.real();
            self_coef_i[p] = sc.imag();
        }
        EF->setSelfCoef(self_coef_r, self_coef_i);

        // Calculate guess dipoles
        if (wavevec_idx == 0) {
            dip_guess = initial_guess;
        } else {
            dip_guess = calc_guess(frame_dip, frame_quad, wavevec_idx);
        }

        // Solve frame tensor
        std::vector<Complex> step_cap(K * 3, 0.0);
        std::vector<Complex> step_dip(num_particles * K * 3, 0.0);
        std::vector<Complex> step_quad(num_quads * K * 5, 0.0);
        compute_tensor(dip_guess, step_cap, step_dip, step_quad);

        // Store wavevector results
        std::copy(step_cap.begin(), step_cap.end(), frame_cap.begin() + wavevec_idx * K * 3);
        std::copy(step_dip.begin(), step_dip.end(), frame_dip.begin() + wavevec_idx * num_particles * K * 3);
        if (solve_quadrupoles && num_quads > 0) {
            std::copy(step_quad.begin(), step_quad.end(), frame_quad.begin() + wavevec_idx * num_quads * K * 5);
        }
    }
    decrease_indent();

    // Store frame average polarizability in class avg_dips
    avg_dips.insert(avg_dips.end(), frame_cap.begin(), frame_cap.end());

    return frame_dip;
}

// -----------------------------------------------------------------------------
// Entry Point compute()
// -----------------------------------------------------------------------------

void Dipole_Solver::compute(const std::vector<double>& x_part,
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

        // Instantiate CUDA-accelerated Electric_Field solver
        if (field_type == "direct") {
            EF = std::make_unique<Direct_Electric_Field>(radius, solve_quadrupoles, quad_idxs);
        } else {
            bool use_polydisperse = false;
            if (field_type == "polydisperse") {
                use_polydisperse = true;
            } else if (field_type == "monodisperse") {
                use_polydisperse = false;
            } else if (field_type == "auto") {
                // Check if radii are identical
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
                EF = std::make_unique<Polydisperse_Ewald_Electric_Field>(box[0], box[1], box[2], tol, xi, true, radius, solve_quadrupoles, quad_idxs);
            } else {
                EF = std::make_unique<Monodisperse_Ewald_Electric_Field>(box[0], box[1], box[2], tol, xi, true, solve_quadrupoles, quad_idxs);
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

    // Set positions inside Electric_Field (also triggers internal coordinate updates and precalcs)
    EF->updateParticleCoordinates(scaled_x, scaled_y, scaled_z);

    // Solve spectrum
    std::vector<Complex> initial_guess;
    size_t K = E0.size();
    if (num_frames == 1) {
        initial_guess = calc_mean_field_guess(0);
    } else {
        // Carry over from previous frame's wavelength 0
        size_t prev_frame_idx = num_frames - 2;
        size_t total_size = num_particles * K * 3 + num_quads * K * 5;
        initial_guess.resize(total_size);

        size_t dip_offset = prev_frame_idx * num_wavevectors * num_particles * K * 3;
        std::copy(dips.begin() + dip_offset, dips.begin() + dip_offset + num_particles * K * 3, initial_guess.begin());

        if (solve_quadrupoles && num_quads > 0) {
            size_t quad_offset = prev_frame_idx * num_wavevectors * num_quads * K * 5;
            std::copy(quads.begin() + quad_offset, quads.begin() + quad_offset + num_quads * K * 5, initial_guess.begin() + num_particles * K * 3);
        }
    }
    std::vector<Complex> frame_quad;
    std::vector<Complex> frame_dip = compute_spectrum(initial_guess, frame_quad);

    // Save solver dipoles in class dips
    dips.insert(dips.end(), frame_dip.begin(), frame_dip.end());
    if (solve_quadrupoles && num_quads > 0) {
        quads.insert(quads.end(), frame_quad.begin(), frame_quad.end());
    }

    if (num_frames != 1) {
        decrease_indent();
        decrease_indent();
    }
}

// -----------------------------------------------------------------------------
// Getters
// -----------------------------------------------------------------------------

std::vector<Complex> Dipole_Solver::get_eff_polarizability(bool physical) const {
    size_t K = E0.size();
    std::vector<Complex> eff_polarizability(num_wavevectors * K * 3, 0.0);
    if (num_frames == 0) return eff_polarizability;

    double factor = physical ? (eps_scale * std::pow(length_scale, 3.0)) : 1.0;
    for (size_t w = 0; w < num_wavevectors; ++w) {
        for (size_t idx = 0; idx < K * 3; ++idx) {
            Complex sum = 0.0;
            for (size_t f = 0; f < num_frames; ++f) {
                sum += avg_dips[f * num_wavevectors * K * 3 + w * K * 3 + idx];
            }
            eff_polarizability[w * K * 3 + idx] = (sum / static_cast<double>(num_frames)) * factor;
        }
    }
    return eff_polarizability;
}

std::vector<Complex> Dipole_Solver::get_dipoles(bool physical) const {
    double factor = physical ? (eps_scale * std::pow(length_scale, 3.0)) : 1.0;
    std::vector<Complex> scaled_dips = dips;
    for (auto& val : scaled_dips) {
        val *= factor;
    }
    return scaled_dips;
}

std::vector<Complex> Dipole_Solver::get_quadrupoles(bool physical) const {
    double factor = physical ? (eps_scale * std::pow(length_scale, 4.0)) : 1.0;
    std::vector<Complex> scaled_quads = quads;
    for (auto& val : scaled_quads) {
        val *= factor;
    }
    return scaled_quads;
}
