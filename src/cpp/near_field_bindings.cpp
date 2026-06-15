#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "near_field.h"

namespace py = pybind11;

void register_near_field(py::module_& m) {
    py::class_<Near_Field>(m, "Near_Field")
        .def(py::init<const std::vector<double>&,
                      const std::vector<Complex>&,
                      const std::vector<double>&,
                      double,
                      double,
                      const std::string&>(),
             py::arg("box"),
             py::arg("E0"),
             py::arg("radius") = std::vector<double>{},
             py::arg("xi") = 0.5,
             py::arg("errortol") = 1e-3,
             py::arg("field_type") = "auto")
        .def("set_dipoles", &Near_Field::set_dipoles, py::arg("dip"))
        .def("set_dipole_positions", &Near_Field::set_dipole_positions, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("set_field_points", &Near_Field::set_field_points, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("calculate", &Near_Field::calculate);
}
