#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include "eels_solver.h"

namespace py = pybind11;

py::array_t<double> get_eels_numpy(const EELS_Solver& self) {
    std::vector<double> vec = self.get_eels();
    size_t num_frames = self.get_num_frames();
    size_t num_waves = self.get_num_wavevectors();
    
    py::array_t<double> result({(ssize_t)num_frames, (ssize_t)num_waves});
    auto buffer = result.request();
    double* ptr = static_cast<double*>(buffer.ptr);
    
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

py::array_t<std::complex<double>> get_dipoles_numpy(const EELS_Solver& self, bool physical = true) {
    std::vector<std::complex<double>> vec = self.get_dipoles(physical);
    size_t num_frames = self.get_num_frames();
    size_t num_waves = self.get_num_wavevectors();
    size_t num_particles = self.get_num_particles();
    
    py::array_t<std::complex<double>> result({
        (ssize_t)num_frames,
        (ssize_t)num_waves,
        (ssize_t)num_particles,
        (ssize_t)3
    });
    auto buffer = result.request();
    std::complex<double>* ptr = static_cast<std::complex<double>*>(buffer.ptr);
    
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

void register_eels(py::module_& m) {
    py::class_<EELS_Solver>(m, "EELS_Solver")
        // 1. 2D eps_p
        .def(py::init<const std::vector<double>&,
                      const std::vector<std::vector<Complex>>&,
                      const std::vector<double>&,
                      double,
                      const std::vector<double>&,
                      double,
                      double,
                      double,
                      bool,
                      const std::string&,
                      const std::string&,
                      const std::string&,
                      double>(),
             py::arg("box"),
             py::arg("eps_p"),
             py::arg("omega"),
             py::arg("v"),
             py::arg("radius") = std::vector<double>{},
             py::arg("eps_m") = 1.0,
             py::arg("xi") = 0.5,
             py::arg("tol") = 1e-3,
             py::arg("quiet") = false,
             py::arg("guess_type") = "derivative",
             py::arg("solver_type") = "gmres",
             py::arg("field_type") = "auto",
             py::arg("integration_step") = 0.01)
        // 2. 1D eps_p
        .def(py::init<const std::vector<double>&,
                      const std::vector<Complex>&,
                      const std::vector<double>&,
                      double,
                      const std::vector<double>&,
                      double,
                      double,
                      double,
                      bool,
                      const std::string&,
                      const std::string&,
                      const std::string&,
                      double>(),
             py::arg("box"),
             py::arg("eps_p_1d"),
             py::arg("omega"),
             py::arg("v"),
             py::arg("radius") = std::vector<double>{},
             py::arg("eps_m") = 1.0,
             py::arg("xi") = 0.5,
             py::arg("tol") = 1e-3,
             py::arg("quiet") = false,
             py::arg("guess_type") = "derivative",
             py::arg("solver_type") = "gmres",
             py::arg("field_type") = "auto",
             py::arg("integration_step") = 0.01)
        // 3. Scalar eps_p
        .def(py::init<const std::vector<double>&,
                      Complex,
                      const std::vector<double>&,
                      double,
                      const std::vector<double>&,
                      double,
                      double,
                      double,
                      bool,
                      const std::string&,
                      const std::string&,
                      const std::string&,
                      double>(),
             py::arg("box"),
             py::arg("eps_p_scalar"),
             py::arg("omega"),
             py::arg("v"),
             py::arg("radius") = std::vector<double>{},
             py::arg("eps_m") = 1.0,
             py::arg("xi") = 0.5,
             py::arg("tol") = 1e-3,
             py::arg("quiet") = false,
             py::arg("guess_type") = "derivative",
             py::arg("solver_type") = "gmres",
             py::arg("field_type") = "auto",
             py::arg("integration_step") = 0.01)
        .def("compute", &EELS_Solver::compute,
             py::arg("epos"),
             py::arg("x_part"),
             py::arg("y_part"),
             py::arg("z_part"))
        .def("get_eels", &get_eels_numpy)
        .def("get_dipoles", &get_dipoles_numpy, py::arg("physical") = true)
        .def("get_num_frames", &EELS_Solver::get_num_frames)
        .def("get_num_particles", &EELS_Solver::get_num_particles)
        .def("get_num_wavevectors", &EELS_Solver::get_num_wavevectors);
}
