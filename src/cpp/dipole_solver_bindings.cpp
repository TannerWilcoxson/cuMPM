#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include "dipole_solver.h"

namespace py = pybind11;

// Helper to convert std::vector<Complex> to a shaped NumPy array for eff_polarizability
py::array_t<std::complex<double>> get_eff_polarizability_numpy(const Dipole_Solver& self) {
    std::vector<std::complex<double>> vec = self.get_eff_polarizability();
    size_t num_waves = self.get_num_wavevectors();
    size_t K = self.get_num_incident_polarizations();
    
    py::array_t<std::complex<double>> result({(ssize_t)num_waves, (ssize_t)K, (ssize_t)3});
    auto buffer = result.request();
    std::complex<double>* ptr = static_cast<std::complex<double>*>(buffer.ptr);
    
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

// Helper to convert std::vector<Complex> to a shaped NumPy array for dipoles
py::array_t<std::complex<double>> get_dipoles_numpy(const Dipole_Solver& self) {
    std::vector<std::complex<double>> vec = self.get_dipoles();
    size_t num_frames = self.get_num_frames();
    size_t num_waves = self.get_num_wavevectors();
    size_t num_particles = self.get_num_particles();
    size_t K = self.get_num_incident_polarizations();
    
    py::array_t<std::complex<double>> result({
        (ssize_t)num_frames,
        (ssize_t)num_waves,
        (ssize_t)num_particles,
        (ssize_t)K,
        (ssize_t)3
    });
    auto buffer = result.request();
    std::complex<double>* ptr = static_cast<std::complex<double>*>(buffer.ptr);
    
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

PYBIND11_MODULE(_cuMPM, m, py::mod_gil_not_used()) {
    m.doc() = "CUDA-accelerated Dipole Solver Python Extension";

    py::class_<Dipole_Solver>(m, "Dipole_Solver")
        .def(py::init<const std::vector<double>&,
                      const std::vector<std::vector<std::complex<double>>>&,
                      const std::vector<double>&,
                      double,
                      double,
                      double,
                      bool,
                      const std::string&,
                      const std::string&,
                      const std::string&,
                      const std::vector<std::vector<std::complex<double>>>&>(),
             py::arg("box"),
             py::arg("eps_p"),
             py::arg("radius") = std::vector<double>{},
             py::arg("eps_m") = 1.0,
             py::arg("xi") = 0.5,
             py::arg("tol") = 1e-3,
             py::arg("quiet") = false,
             py::arg("guess_type") = "derivative",
             py::arg("solver_type") = "gmres",
             py::arg("field_type") = "auto",
             py::arg("E0") = std::vector<std::vector<std::complex<double>>>{})
        .def(py::init<const std::vector<double>&,
                      const std::vector<std::complex<double>>&,
                      const std::vector<double>&,
                      double,
                      double,
                      double,
                      bool,
                      const std::string&,
                      const std::string&,
                      const std::string&,
                      const std::vector<std::vector<std::complex<double>>>&>(),
             py::arg("box"),
             py::arg("eps_p_1d"),
             py::arg("radius") = std::vector<double>{},
             py::arg("eps_m") = 1.0,
             py::arg("xi") = 0.5,
             py::arg("tol") = 1e-3,
             py::arg("quiet") = false,
             py::arg("guess_type") = "derivative",
             py::arg("solver_type") = "gmres",
             py::arg("field_type") = "auto",
             py::arg("E0") = std::vector<std::vector<std::complex<double>>>{})
        .def(py::init<const std::vector<double>&,
                      std::complex<double>,
                      const std::vector<double>&,
                      double,
                      double,
                      double,
                      bool,
                      const std::string&,
                      const std::string&,
                      const std::string&,
                      const std::vector<std::vector<std::complex<double>>>&>(),
             py::arg("box"),
             py::arg("eps_p_scalar"),
             py::arg("radius") = std::vector<double>{},
             py::arg("eps_m") = 1.0,
             py::arg("xi") = 0.5,
             py::arg("tol") = 1e-3,
             py::arg("quiet") = false,
             py::arg("guess_type") = "derivative",
             py::arg("solver_type") = "gmres",
             py::arg("field_type") = "auto",
             py::arg("E0") = std::vector<std::vector<std::complex<double>>>{})
        .def("compute", &Dipole_Solver::compute,
             py::arg("x_part"),
             py::arg("y_part"),
             py::arg("z_part"))
        .def("get_eff_polarizability", &get_eff_polarizability_numpy)
        .def("get_dipoles", &get_dipoles_numpy)
        .def("get_num_frames", &Dipole_Solver::get_num_frames)
        .def("get_num_particles", &Dipole_Solver::get_num_particles)
        .def("get_num_wavevectors", &Dipole_Solver::get_num_wavevectors)
        .def("get_num_incident_polarizations", &Dipole_Solver::get_num_incident_polarizations);

    // Forward declaration of register helper
    void register_near_field(py::module_& m);
    register_near_field(m);
}
