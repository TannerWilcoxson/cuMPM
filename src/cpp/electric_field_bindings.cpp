#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <optional>
#include <cstdint>
#include "electric_field/electric_field.h"
#include "electric_field/direct_electric_field.h"
#include "electric_field/ewald_electric_field_base.h"
#include "electric_field/monodisperse_ewald_electric_field.h"
#include "electric_field/polydisperse_ewald_electric_field.h"

namespace py = pybind11;

void register_electric_fields(py::module_& m) {
    py::enum_<FieldCalcMode>(m, "FieldCalcMode")
        .value("SOLVER_AX", FieldCalcMode::SOLVER_AX)
        .value("INTERACTION_FIELD", FieldCalcMode::INTERACTION_FIELD)
        .export_values();

    py::enum_<PrecisionMode>(m, "PrecisionMode")
        .value("AUTO", PrecisionMode::AUTO)
        .value("MIXED", PrecisionMode::MIXED)
        .value("DOUBLE", PrecisionMode::DOUBLE)
        .value("FP32", PrecisionMode::FP32)
        .value("FP64", PrecisionMode::FP64)
        .export_values();

    // Expose abstract base class Electric_Field
    py::class_<Electric_Field, std::unique_ptr<Electric_Field>>(m, "Electric_Field")
        .def("getDevDipoles", [](const Electric_Field& self) {
            return reinterpret_cast<uintptr_t>(self.getDevDipoles());
        })
        .def("getDevEPoint", [](const Electric_Field& self) {
            return reinterpret_cast<uintptr_t>(self.getDevEPoint());
        })
        .def("calculate", &Electric_Field::calculate)
        .def("updateParticleCoordinates", &Electric_Field::updateParticleCoordinates,
             py::arg("x_part"), py::arg("y_part"), py::arg("z_part"))
        .def("setSelfCoef", py::overload_cast<const std::vector<std::complex<double>>&>(&Electric_Field::setSelfCoef), py::arg("self_coef"))
        .def("setSelfCoef", py::overload_cast<const std::vector<double>&, const std::vector<double>&>(&Electric_Field::setSelfCoef), py::arg("self_coef_r"), py::arg("self_coef_i"));

    // Expose Direct_Electric_Field
    py::class_<Direct_Electric_Field, Electric_Field, std::unique_ptr<Direct_Electric_Field>>(m, "Direct_Electric_Field")
        .def(py::init<const std::vector<double>&, FieldCalcMode, bool, const std::vector<int>&, PrecisionMode>(),
             py::arg("radius") = std::vector<double>{},
             py::arg("mode") = FieldCalcMode::SOLVER_AX,
             py::arg("solve_quadrupoles") = false,
             py::arg("quad_idxs") = std::vector<int>{},
             py::arg("precision") = PrecisionMode::AUTO)
        .def("isUsingFP32", &Direct_Electric_Field::isUsingFP32)
        .def("getPrecisionMode", &Direct_Electric_Field::getPrecisionMode)
        .def("getCalcMode", &Direct_Electric_Field::getCalcMode)
        .def("getSolveQuadrupoles", &Direct_Electric_Field::getSolveQuadrupoles)
        .def("getNumQuads", &Direct_Electric_Field::getNumQuads)
        .def("getNumFieldPoints", &Direct_Electric_Field::getNumFieldPoints)
        .def("getNumParticles", &Direct_Electric_Field::getNumParticles)
        .def("getEPointHost", &Direct_Electric_Field::getEPointHost)
        .def("updateFieldCoordinates", &Direct_Electric_Field::updateFieldCoordinates,
             py::arg("x_field"), py::arg("y_field"), py::arg("z_field"))
        .def("getParticlesUpdated", &Direct_Electric_Field::getParticlesUpdated)
        .def("getFieldPointsUpdated", &Direct_Electric_Field::getFieldPointsUpdated)
        .def("clearParticlesUpdated", &Direct_Electric_Field::clearParticlesUpdated)
        .def("clearFieldPointsUpdated", &Direct_Electric_Field::clearFieldPointsUpdated)
        .def("getDipolesUpdated", &Direct_Electric_Field::getDipolesUpdated)
        .def("clearDipolesUpdated", &Direct_Electric_Field::clearDipolesUpdated)
        .def("getDevXPart", [](const Direct_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevXPart()); })
        .def("getDevYPart", [](const Direct_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevYPart()); })
        .def("getDevZPart", [](const Direct_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevZPart()); })
        .def("getDevXField", [](const Direct_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevXField()); })
        .def("getDevYField", [](const Direct_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevYField()); })
        .def("getDevZField", [](const Direct_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevZField()); })
        .def("getDevSelfCoef", [](const Direct_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevSelfCoef()); })
        .def("updateDipoles", &Direct_Electric_Field::updateDipoles,
             py::arg("dip_x"), py::arg("dip_y"), py::arg("dip_z"))
        .def("updateDipolesComplex", &Direct_Electric_Field::updateDipolesComplex,
             py::arg("dip_xr"), py::arg("dip_xi"), py::arg("dip_yr"), py::arg("dip_yi"), py::arg("dip_zr"), py::arg("dip_zi"))
        .def("updateQuadrupoles", &Direct_Electric_Field::updateQuadrupoles,
             py::arg("quad_1"), py::arg("quad_2"), py::arg("quad_3"), py::arg("quad_4"), py::arg("quad_5"))
        .def("updateQuadrupolesComplex", &Direct_Electric_Field::updateQuadrupolesComplex,
             py::arg("quad_1r"), py::arg("quad_1i"), py::arg("quad_2r"), py::arg("quad_2i"),
             py::arg("quad_3r"), py::arg("quad_3i"), py::arg("quad_4r"), py::arg("quad_4i"),
             py::arg("quad_5r"), py::arg("quad_5i"))
        .def("getDipolesHost", [](const Direct_Electric_Field& self) {
            std::vector<double> host_dip_x;
            std::vector<double> host_dip_y;
            std::vector<double> host_dip_z;
            self.getDipolesHost(host_dip_x, host_dip_y, host_dip_z);
            return std::make_tuple(host_dip_x, host_dip_y, host_dip_z);
        })
        .def("getDipolesComplexHost", [](const Direct_Electric_Field& self) {
            std::vector<double> host_dip_xr; std::vector<double> host_dip_xi;
            std::vector<double> host_dip_yr; std::vector<double> host_dip_yi;
            std::vector<double> host_dip_zr; std::vector<double> host_dip_zi;
            self.getDipolesComplexHost(host_dip_xr, host_dip_xi, host_dip_yr, host_dip_yi, host_dip_zr, host_dip_zi);
            return std::make_tuple(host_dip_xr, host_dip_xi, host_dip_yr, host_dip_yi, host_dip_zr, host_dip_zi);
        });

    // Expose Ewald_Electric_Field_Base
    py::class_<Ewald_Electric_Field_Base, Electric_Field, std::unique_ptr<Ewald_Electric_Field_Base>>(m, "Ewald_Electric_Field_Base")
        // Scalar and state getters/setters
        .def("getBoxX", &Ewald_Electric_Field_Base::getBoxX)
        .def("getBoxY", &Ewald_Electric_Field_Base::getBoxY)
        .def("getBoxZ", &Ewald_Electric_Field_Base::getBoxZ)
        .def("getErrortol", &Ewald_Electric_Field_Base::getErrortol)
        .def("getRc", &Ewald_Electric_Field_Base::getRc)
        .def("getXi", &Ewald_Electric_Field_Base::getXi)
        .def("getCalcMode", &Ewald_Electric_Field_Base::getCalcMode)
        .def("getSelfCoef", &Ewald_Electric_Field_Base::getSelfCoef)
        .def("getNumParticles", &Ewald_Electric_Field_Base::getNumParticles)
        .def("getNumFieldPoints", &Ewald_Electric_Field_Base::getNumFieldPoints)
        .def("getParticlesUpdated", &Ewald_Electric_Field_Base::getParticlesUpdated)
        .def("getFieldPointsUpdated", &Ewald_Electric_Field_Base::getFieldPointsUpdated)
        .def("clearParticlesUpdated", &Ewald_Electric_Field_Base::clearParticlesUpdated)
        .def("clearFieldPointsUpdated", &Ewald_Electric_Field_Base::clearFieldPointsUpdated)
        .def("getDipolesUpdated", &Ewald_Electric_Field_Base::getDipolesUpdated)
        .def("clearDipolesUpdated", &Ewald_Electric_Field_Base::clearDipolesUpdated)
        .def("getNumSpread", &Ewald_Electric_Field_Base::getNumSpread)
        .def("getNumContract", &Ewald_Electric_Field_Base::getNumContract)
        .def("getNumPairs", &Ewald_Electric_Field_Base::getNumPairs)
        .def("getMaxNeighbors", &Ewald_Electric_Field_Base::getMaxNeighbors)
        .def("getTableSize", &Ewald_Electric_Field_Base::getTableSize)
        .def("getNumOffsets", &Ewald_Electric_Field_Base::getNumOffsets)
        .def("getSolveQuadrupoles", &Ewald_Electric_Field_Base::getSolveQuadrupoles)
        .def("getNumQuads", &Ewald_Electric_Field_Base::getNumQuads)
        .def("isUsingAsyncStreams", &Ewald_Electric_Field_Base::isUsingAsyncStreams)
        .def("setUseAsyncStreams", &Ewald_Electric_Field_Base::setUseAsyncStreams, py::arg("enable"))
        .def("setBlochWavevector", &Ewald_Electric_Field_Base::setBlochWavevector, py::arg("kx"), py::arg("ky"))
        // Array getters
        .def("getNumGrid", [](const Ewald_Electric_Field_Base& self) {
            const int* grid = self.getNumGrid();
            return std::vector<int>{grid[0], grid[1], grid[2]};
        })
        .def("getGridSpacing", [](const Ewald_Electric_Field_Base& self) {
            const double* spacing = self.getGridSpacing();
            return std::vector<double>{spacing[0], spacing[1], spacing[2]};
        })
        .def("getSpectralSplit", [](const Ewald_Electric_Field_Base& self) {
            const double* split = self.getSpectralSplit();
            return std::vector<double>{split[0], split[1], split[2]};
        })
        // Device pointer getters
        .def("getDevXPart", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevXPart()); })
        .def("getDevYPart", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevYPart()); })
        .def("getDevZPart", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevZPart()); })
        .def("getDevXField", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevXField()); })
        .def("getDevYField", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevYField()); })
        .def("getDevZField", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevZField()); })
        .def("getDevSelfCoef", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevSelfCoef()); })
        .def("getDevSpreadCoef", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevSpreadCoef()); })
        .def("getDevSpreadIdxs", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevSpreadIdxs()); })
        .def("getDevParticleIndex", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevParticleIndex()); })
        .def("getDevContractCoef", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevContractCoef()); })
        .def("getDevContractIdxs", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevContractIdxs()); })
        .def("getDevPerp", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevPerp()); })
        .def("getDevPara", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevPara()); })
        .def("getDevFEGrid", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevFEGrid()); })
        .def("getDevNeighborList", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevNeighborList()); })
        .def("getDevNeighborCounts", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevNeighborCounts()); })
        .def("getDevRTable", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevRTable()); })
        .def("getDevFieldDip1", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevFieldDip1()); })
        .def("getDevFieldDip2", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevFieldDip2()); })
        .def("getDevOffset", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevOffset()); })
        .def("getDevOffsetxyz", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevOffsetxyz()); })
        .def("getDevScaleCoef", [](const Ewald_Electric_Field_Base& self) { return reinterpret_cast<uintptr_t>(self.getDevScaleCoef()); })
        // Action methods
        .def("computeNeighborList", &Ewald_Electric_Field_Base::computeNeighborList, py::arg("max_neighbors_per_particle") = 128)
        .def("updateFieldCoordinates", &Ewald_Electric_Field_Base::updateFieldCoordinates,
             py::arg("x_field"), py::arg("y_field"), py::arg("z_field"))
        .def("updateDipoles", &Ewald_Electric_Field_Base::updateDipoles,
             py::arg("dip_x"), py::arg("dip_y"), py::arg("dip_z"))
        .def("updateDipolesComplex", &Ewald_Electric_Field_Base::updateDipolesComplex,
             py::arg("dip_xr"), py::arg("dip_xi"), py::arg("dip_yr"), py::arg("dip_yi"), py::arg("dip_zr"), py::arg("dip_zi"))
        .def("updateQuadrupoles", &Ewald_Electric_Field_Base::updateQuadrupoles,
             py::arg("quad_1"), py::arg("quad_2"), py::arg("quad_3"), py::arg("quad_4"), py::arg("quad_5"))
        .def("updateQuadrupolesComplex", &Ewald_Electric_Field_Base::updateQuadrupolesComplex,
             py::arg("quad_1r"), py::arg("quad_1i"), py::arg("quad_2r"), py::arg("quad_2i"),
             py::arg("quad_3r"), py::arg("quad_3i"), py::arg("quad_4r"), py::arg("quad_4i"),
             py::arg("quad_5r"), py::arg("quad_5i"))
        .def("setSelfCoef", py::overload_cast<const std::vector<double>&, const std::vector<double>&>(&Ewald_Electric_Field_Base::setSelfCoef),
             py::arg("self_coef_r"), py::arg("self_coef_i"))
        .def("setSelfCoef", py::overload_cast<double, double>(&Ewald_Electric_Field_Base::setSelfCoef),
             py::arg("val_r"), py::arg("val_i") = 0.0)
        .def("spreadPrecalcs", &Ewald_Electric_Field_Base::spreadPrecalcs)
        .def("contractPrecalcs", &Ewald_Electric_Field_Base::contractPrecalcs)
        .def("realSpacePrecalcs", &Ewald_Electric_Field_Base::realSpacePrecalcs)
        // Host data-copy verification helper methods
        .def("getNeighborListHost", [](const Ewald_Electric_Field_Base& self) {
            std::vector<int> host_list;
            std::vector<int> host_counts;
            self.getNeighborListHost(host_list, host_counts);
            return std::make_pair(host_list, host_counts);
        })
        .def("getRealSpaceTablesHost", [](const Ewald_Electric_Field_Base& self) {
            std::vector<double> host_r_table;
            std::vector<double> host_field_dip_1;
            std::vector<double> host_field_dip_2;
            self.getRealSpaceTablesHost(host_r_table, host_field_dip_1, host_field_dip_2);
            return std::make_tuple(host_r_table, host_field_dip_1, host_field_dip_2);
        })
        .def("getDipolesHost", [](const Ewald_Electric_Field_Base& self) {
            std::vector<double> host_dip_x;
            std::vector<double> host_dip_y;
            std::vector<double> host_dip_z;
            self.getDipolesHost(host_dip_x, host_dip_y, host_dip_z);
            return std::make_tuple(host_dip_x, host_dip_y, host_dip_z);
        })
        .def("getDipolesComplexHost", [](const Ewald_Electric_Field_Base& self) {
            std::vector<double> host_dip_xr; std::vector<double> host_dip_xi;
            std::vector<double> host_dip_yr; std::vector<double> host_dip_yi;
            std::vector<double> host_dip_zr; std::vector<double> host_dip_zi;
            self.getDipolesComplexHost(host_dip_xr, host_dip_xi, host_dip_yr, host_dip_yi, host_dip_zr, host_dip_zi);
            return std::make_tuple(host_dip_xr, host_dip_xi, host_dip_yr, host_dip_yi, host_dip_zr, host_dip_zi);
        })
        .def("getSpreadPrecalcsHost", [](const Ewald_Electric_Field_Base& self) {
            std::vector<double> host_spread_coef;
            std::vector<int> host_spread_idxs;
            self.getSpreadPrecalcsHost(host_spread_coef, host_spread_idxs);
            return std::make_pair(host_spread_coef, host_spread_idxs);
        })
        .def("getContractPrecalcsHost", [](const Ewald_Electric_Field_Base& self) {
            std::vector<double> host_E_point;
            std::vector<int> host_particle_index;
            std::vector<double> host_contract_coef;
            std::vector<int> host_contract_idxs;
            self.getContractPrecalcsHost(host_E_point, host_particle_index, host_contract_coef, host_contract_idxs);
            return std::make_tuple(host_E_point, host_particle_index, host_contract_coef, host_contract_idxs);
        })
        .def("getRealSpacePrecalcsHost", [](const Ewald_Electric_Field_Base& self) {
            double host_self_perp = 0.0;
            std::vector<double> host_perp;
            std::vector<double> host_para;
            self.getRealSpacePrecalcsHost(host_self_perp, host_perp, host_para);
            return std::make_tuple(host_self_perp, host_perp, host_para);
        })
        .def("getEPointHost", &Ewald_Electric_Field_Base::getEPointHost)
        .def("isUsingRecipFP32", &Ewald_Electric_Field_Base::isUsingRecipFP32)
        .def("getRecipPrecisionMode", &Ewald_Electric_Field_Base::getRecipPrecisionMode)
        // Pipeline operations with optional custom GPU pointer support
        .def("spread", [](Ewald_Electric_Field_Base& self, std::optional<uintptr_t> grid_ptr) {
            void* ptr = grid_ptr ? reinterpret_cast<void*>(*grid_ptr) : self.getDevFEGrid();
            self.spread(ptr);
        }, py::arg("grid_ptr") = py::none())
        .def("scale", [](Ewald_Electric_Field_Base& self, std::optional<uintptr_t> grid_ptr) {
            void* ptr = grid_ptr ? reinterpret_cast<void*>(*grid_ptr) : self.getDevFEGrid();
            self.scale(ptr);
        }, py::arg("grid_ptr") = py::none())
        .def("contract", [](Ewald_Electric_Field_Base& self, std::optional<uintptr_t> E_point_ptr, std::optional<uintptr_t> Es_grid_ptr) {
            double2* E_ptr = E_point_ptr ? reinterpret_cast<double2*>(*E_point_ptr) : self.getDevEPoint();
            const void* Es_ptr = Es_grid_ptr ? reinterpret_cast<const void*>(*Es_grid_ptr) : self.getDevFEGrid();
            self.contract(E_ptr, Es_ptr);
        }, py::arg("E_point_ptr") = py::none(), py::arg("Es_grid_ptr") = py::none())
        .def("realSpace", [](Ewald_Electric_Field_Base& self, std::optional<uintptr_t> E_point_ptr) {
            double2* ptr = E_point_ptr ? reinterpret_cast<double2*>(*E_point_ptr) : self.getDevEPoint();
            self.realSpace(ptr);
        }, py::arg("E_point_ptr") = py::none())
        .def("clearEPoint", &Ewald_Electric_Field_Base::clearEPoint);

    // Expose Monodisperse_Ewald_Electric_Field
    py::class_<Monodisperse_Ewald_Electric_Field, Ewald_Electric_Field_Base, std::unique_ptr<Monodisperse_Ewald_Electric_Field>>(m, "Monodisperse_Ewald_Electric_Field")
        .def(py::init<double, double, double, double, double, FieldCalcMode, double, bool, const std::vector<int>&, PrecisionMode>(),
             py::arg("box_x"), py::arg("box_y"), py::arg("box_z"),
             py::arg("errortol"), py::arg("xi"), py::arg("mode"),
             py::arg("radius") = 1.0,
             py::arg("solve_quadrupoles") = false,
             py::arg("quad_idxs") = std::vector<int>{},
             py::arg("recip_precision") = PrecisionMode::AUTO)
        .def("getDevSelfPerp", [](const Monodisperse_Ewald_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevSelfPerp()); })
        .def("getDevKhat", [](const Monodisperse_Ewald_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevKhat()); })
        .def("computePrecalculations", &Monodisperse_Ewald_Electric_Field::computePrecalculations)
        .def("computeRealSpaceTables", &Monodisperse_Ewald_Electric_Field::computeRealSpaceTables)
        .def("getPrecalculationsHost", [](const Monodisperse_Ewald_Electric_Field& self) {
            std::vector<int> host_offset;
            std::vector<double> host_offsetxyz;
            std::vector<double> host_scale_coef;
            std::vector<double> host_khat;
            self.getPrecalculationsHost(host_offset, host_offsetxyz, host_scale_coef, host_khat);
            return std::make_tuple(host_offset, host_offsetxyz, host_scale_coef, host_khat);
        });

    // Expose Polydisperse_Ewald_Electric_Field
    py::class_<Polydisperse_Ewald_Electric_Field, Ewald_Electric_Field_Base, std::unique_ptr<Polydisperse_Ewald_Electric_Field>>(m, "Polydisperse_Ewald_Electric_Field")
        .def(py::init<double, double, double, double, double, FieldCalcMode, const std::vector<double>&, bool, const std::vector<int>&, PrecisionMode>(),
             py::arg("box_x"), py::arg("box_y"), py::arg("box_z"),
             py::arg("errortol"), py::arg("xi"), py::arg("mode"),
             py::arg("particle_radii"),
             py::arg("solve_quadrupoles") = false,
             py::arg("quad_idxs") = std::vector<int>{},
             py::arg("recip_precision") = PrecisionMode::AUTO)
        .def("getEta", &Polydisperse_Ewald_Electric_Field::getEta)
        .def("getP", &Polydisperse_Ewald_Electric_Field::getP)
        .def("getDevGPoint", [](const Polydisperse_Ewald_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevGPoint()); })
        .def("getDevQuadIdxs", [](const Polydisperse_Ewald_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevQuadIdxs()); })
        .def("getDevQuadMap", [](const Polydisperse_Ewald_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevQuadMap()); })
        .def("getDevSpreadCoefQ", [](const Polydisperse_Ewald_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevSpreadCoefQ()); })
        .def("getDevContractCoefQ", [](const Polydisperse_Ewald_Electric_Field& self) { return reinterpret_cast<uintptr_t>(self.getDevContractCoefQ()); })
        .def("computeRealSpaceTables", &Polydisperse_Ewald_Electric_Field::computeRealSpaceTables)
        .def("computePrecalculations", &Polydisperse_Ewald_Electric_Field::computePrecalculations)
        .def("getPrecalculationsHost", [](const Polydisperse_Ewald_Electric_Field& self) {
            std::vector<int> host_offset;
            std::vector<double> host_offsetxyz;
            std::vector<double> host_scale_coef;
            self.getPrecalculationsHost(host_offset, host_offsetxyz, host_scale_coef);
            return std::make_tuple(host_offset, host_offsetxyz, host_scale_coef);
        });
}
