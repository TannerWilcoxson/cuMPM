#ifndef DIPOLE_SOLVER_H
#define DIPOLE_SOLVER_H

#include <vector>
#include <complex>
#include <string>
#include <memory>
#include "electric_field.h"

using Complex = std::complex<double>;

class Dipole_Solver {
private:
    std::vector<double> box;
    std::vector<std::vector<Complex>> eps_p; // shape: num_wavevectors x num_particles
    std::vector<double> radius;              // shape: num_particles
    double eps_m;
    double xi;
    double tol;
    bool quiet;
    std::string guess_type;

    size_t num_particles = 0;
    size_t num_wavevectors = 0;
    size_t num_frames = 0;
    double length_scale = 1.0;
    double eps_scale = 1.0;
    double vol_frac = 0.0;

    std::unique_ptr<Electric_Field> EF;

    // GPU buffers for the resident solver
    double* d_x = nullptr;
    double* d_b = nullptr;
    double* d_r = nullptr;
    double* d_w = nullptr;
    double* d_V = nullptr; // Krylov basis vectors [restart + 1]
    double* d_reduce_buf = nullptr; // shared helper buffer for reductions (norm, dot)

    // Results storage
    // avg_dips: average polarizability tensor over all frames for each wavelength
    // shape: num_wavevectors * 3 * 3 (layout: wavevec * 9)
    std::vector<Complex> avg_dips;
    // dips: calculated dipoles for all frames, wavelengths, particles
    // shape: num_frames * num_wavevectors * num_particles * 3 * 3 (layout: frame * wavevec * part * 9)
    std::vector<Complex> dips;

    // Helper functions
    void set_dims(size_t num_p);
    void nondimensionalize();
    void precalculations();
    void calc_vol_frac();

    // Print utility
    int indent_level = 0;
    void print(const std::string& msg) const;
    void increase_indent() { indent_level++; }
    void decrease_indent() { if (indent_level > 0) indent_level--; }

    // Guess calculation
    std::vector<Complex> calc_guess(const std::vector<Complex>& prev_dip, size_t wavevec_idx) const;
    std::vector<Complex> calc_mean_field_guess(size_t wavevec_idx) const;
    std::vector<Complex> calc_previous_guess(const std::vector<Complex>& prev_dip, size_t wavevec_idx) const;
    std::vector<Complex> calc_derivative_guess(const std::vector<Complex>& prev_dip, size_t wavevec_idx) const;

    // GMRES and system solve (GPU resident)
    void allocate_gpu_buffers();
    void free_gpu_buffers();
    std::vector<Complex> compute_spectrum(const std::vector<Complex>& initial_guess);
    void compute_tensor(const std::vector<Complex>& dip_guess, 
                        std::vector<Complex>& frame_cap, 
                        std::vector<Complex>& frame_dip);
    std::vector<Complex> compute_dipoles(const std::vector<Complex>& E, 
                                         const std::vector<Complex>& dip_guess);

public:
    // Constructors
    // 1. Full 2D eps_p
    Dipole_Solver(const std::vector<double>& box,
                  const std::vector<std::vector<Complex>>& eps_p,
                  const std::vector<double>& radius = {},
                  double eps_m = 1.0,
                  double xi = 0.5,
                  double tol = 1e-3,
                  bool quiet = false,
                  const std::string& guess_type = "derivative");

    // 2. 1D eps_p (wavelength-dependent, same for all particles)
    Dipole_Solver(const std::vector<double>& box,
                  const std::vector<Complex>& eps_p_1d,
                  const std::vector<double>& radius = {},
                  double eps_m = 1.0,
                  double xi = 0.5,
                  double tol = 1e-3,
                  bool quiet = false,
                  const std::string& guess_type = "derivative");

    // 3. Scalar eps_p (single wavelength, same for all particles)
    Dipole_Solver(const std::vector<double>& box,
                  Complex eps_p_scalar,
                  const std::vector<double>& radius = {},
                  double eps_m = 1.0,
                  double xi = 0.5,
                  double tol = 1e-3,
                  bool quiet = false,
                  const std::string& guess_type = "derivative");

    ~Dipole_Solver();

    // Main solver entry point
    void compute(const std::vector<double>& x_part,
                 const std::vector<double>& y_part,
                 const std::vector<double>& z_part);

    // Getters for results
    // Returns average effective polarizability over all processed frames
    // shape: num_wavevectors * 3 * 3
    std::vector<Complex> get_eff_polarizability() const;

    // Returns all calculated dipoles
    // shape: num_frames * num_wavevectors * num_particles * 3 * 3
    std::vector<Complex> get_dipoles() const;

    size_t get_num_frames() const { return num_frames; }
    size_t get_num_particles() const { return num_particles; }
    size_t get_num_wavevectors() const { return num_wavevectors; }
};

#endif // DIPOLE_SOLVER_H
