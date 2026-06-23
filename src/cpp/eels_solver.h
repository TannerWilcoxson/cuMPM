#ifndef EELS_SOLVER_H
#define EELS_SOLVER_H

#include <vector>
#include <complex>
#include <string>
#include <memory>
#include "electric_field/electric_field.h"
#include "numerical_solver/numerical_solver.h"

using Complex = std::complex<double>;

class EELS_Solver {
private:
    std::vector<double> box;
    std::vector<std::vector<Complex>> eps_p; // shape: num_wavevectors x num_particles
    std::vector<double> omega;               // shape: num_wavevectors (wavenumbers)
    double v;                                // electron velocity fraction of c
    std::vector<double> radius;              // shape: num_particles
    double eps_m;
    double xi;
    double tol;
    bool quiet;
    std::string guess_type;
    std::string solver_type;
    std::string field_type;

    size_t num_particles = 0;
    size_t num_wavevectors = 0;
    size_t num_frames = 0;
    double length_scale = 1.0;
    double eps_scale = 1.0;

    std::unique_ptr<Electric_Field> EF;
    std::unique_ptr<Electric_Field> eels_EF;
    std::unique_ptr<Numerical_Solver> solver;

    // Simpson's rule integration coordinates along electron path
    std::vector<double> z_pts;
    std::vector<double> Z_pts;

    // Results storage
    // eels: computed eels probability spectrum for each frame and wavelength
    // shape: num_frames * num_wavevectors
    std::vector<double> eels;
    // dips: calculated particle dipoles for each frame, wavelength, and particle
    // shape: num_frames * num_wavevectors * num_particles * 3
    std::vector<Complex> dips;

    // Helper functions
    void set_dims(size_t num_p);
    void nondimensionalize();
    void precalculations();
    void precomp_eels(const std::vector<double>& z_part);

    // Print utilities
    int indent_level = 0;
    void print(const std::string& msg) const;
    void increase_indent() { indent_level++; }
    void decrease_indent() { if (indent_level > 0) indent_level--; }

    // Guess calculation
    std::vector<Complex> calc_guess(const std::vector<Complex>& prev_dip, size_t wavevec_idx, const std::vector<Complex>& E_inc) const;
    std::vector<Complex> calc_mean_field_guess(size_t wavevec_idx, const std::vector<Complex>& E) const;
    std::vector<Complex> calc_previous_guess(const std::vector<Complex>& prev_dip, size_t wavevec_idx) const;
    std::vector<Complex> calc_derivative_guess(const std::vector<Complex>& prev_dip, size_t wavevec_idx) const;

public:
    void set_numerical_solver(std::unique_ptr<Numerical_Solver> new_solver) {
        solver = std::move(new_solver);
    }

    // Constructors
    // 1. Full 2D eps_p
    EELS_Solver(const std::vector<double>& box,
                const std::vector<std::vector<Complex>>& eps_p,
                const std::vector<double>& omega,
                double v,
                const std::vector<double>& radius = {},
                double eps_m = 1.0,
                double xi = 0.5,
                double tol = 1e-3,
                bool quiet = false,
                const std::string& guess_type = "derivative",
                const std::string& solver_type = "gmres",
                const std::string& field_type = "auto");

    // 2. 1D eps_p
    EELS_Solver(const std::vector<double>& box,
                const std::vector<Complex>& eps_p_1d,
                const std::vector<double>& omega,
                double v,
                const std::vector<double>& radius = {},
                double eps_m = 1.0,
                double xi = 0.5,
                double tol = 1e-3,
                bool quiet = false,
                const std::string& guess_type = "derivative",
                const std::string& solver_type = "gmres",
                const std::string& field_type = "auto");

    // 3. Scalar eps_p
    EELS_Solver(const std::vector<double>& box,
                Complex eps_p_scalar,
                const std::vector<double>& omega,
                double v,
                const std::vector<double>& radius = {},
                double eps_m = 1.0,
                double xi = 0.5,
                double tol = 1e-3,
                bool quiet = false,
                const std::string& guess_type = "derivative",
                const std::string& solver_type = "gmres",
                const std::string& field_type = "auto");

    ~EELS_Solver();

    void compute(const std::vector<double>& epos,
                 const std::vector<double>& x_part,
                 const std::vector<double>& y_part,
                 const std::vector<double>& z_part);

    std::vector<double> get_eels() const { return eels; }
    std::vector<Complex> get_dipoles() const { return dips; }
    size_t get_num_frames() const { return num_frames; }
    size_t get_num_particles() const { return num_particles; }
    size_t get_num_wavevectors() const { return num_wavevectors; }
};

#endif // EELS_SOLVER_H
