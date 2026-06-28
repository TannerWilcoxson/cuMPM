import numpy as np
from .. import _cuMPM

def _to_list(val):
    if val is None:
        return []
    if isinstance(val, (np.ndarray, list, tuple)):
        if isinstance(val, np.ndarray):
            return val.tolist()
        return list(val)
    return [val]

class DirectElectricField:
    """
    Python wrapper for the C++ Direct_Electric_Field class.
    """
    def __init__(self, radius=None, solve_quadrupoles=False, quad_idxs=None):
        if radius is None:
            radius = []
        if quad_idxs is None:
            quad_idxs = []
        self._ef = _cuMPM.Direct_Electric_Field(
            _to_list(radius), bool(solve_quadrupoles), [int(i) for i in quad_idxs]
        )

    @property
    def box_x(self) -> float:
        return 0.0

    @property
    def box_y(self) -> float:
        return 0.0

    @property
    def box_z(self) -> float:
        return 0.0

    @property
    def errortol(self) -> float:
        return 0.0

    @property
    def rc(self) -> float:
        return 0.0

    @property
    def xi(self) -> float:
        return 0.0

    @property
    def calc_inter_dipole(self) -> bool:
        return False

    @property
    def self_coef(self) -> float:
        return 0.0

    @property
    def num_particles(self) -> int:
        return self._ef.getNumParticles()

    @property
    def num_field_points(self) -> int:
        return self._ef.getNumFieldPoints()

    @property
    def particles_updated(self) -> bool:
        return self._ef.getParticlesUpdated()

    @property
    def field_points_updated(self) -> bool:
        return self._ef.getFieldPointsUpdated()

    @property
    def dipoles_updated(self) -> bool:
        return self._ef.getDipolesUpdated()

    @property
    def num_spread(self) -> int:
        return 0

    @property
    def num_contract(self) -> int:
        return 0

    @property
    def num_pairs(self) -> int:
        return 0

    @property
    def max_neighbors(self) -> int:
        return 0

    @property
    def table_size(self) -> int:
        return 0

    @property
    def num_offsets(self) -> int:
        return 0

    @property
    def solve_quadrupoles(self) -> bool:
        return self._ef.getSolveQuadrupoles()

    @property
    def num_quads(self) -> int:
        return self._ef.getNumQuads()

    @property
    def num_grid(self) -> np.ndarray:
        return np.zeros(3, dtype=np.int32)

    @property
    def grid_spacing(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    @property
    def spectral_split(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    # Device pointers
    @property
    def dev_dipoles(self) -> int:
        return self._ef.getDevDipoles()

    @property
    def dev_epoint(self) -> int:
        return self._ef.getDevEPoint()

    @property
    def dev_x_part(self) -> int:
        return self._ef.getDevXPart()

    @property
    def dev_y_part(self) -> int:
        return self._ef.getDevYPart()

    @property
    def dev_z_part(self) -> int:
        return self._ef.getDevZPart()

    @property
    def dev_x_field(self) -> int:
        return self._ef.getDevXField()

    @property
    def dev_y_field(self) -> int:
        return self._ef.getDevYField()

    @property
    def dev_z_field(self) -> int:
        return self._ef.getDevZField()

    @property
    def dev_self_coef_real(self) -> int:
        return self._ef.getDevSelfCoefReal()

    @property
    def dev_self_coef_imag(self) -> int:
        return self._ef.getDevSelfCoefImag()

    @property
    def dev_spread_coef(self) -> int:
        return 0

    @property
    def dev_spread_idxs(self) -> int:
        return 0

    @property
    def dev_particle_index(self) -> int:
        return 0

    @property
    def dev_contract_coef(self) -> int:
        return 0

    @property
    def dev_contract_idxs(self) -> int:
        return 0

    @property
    def dev_self_perp(self) -> int:
        return 0

    @property
    def dev_perp(self) -> int:
        return 0

    @property
    def dev_para(self) -> int:
        return 0

    @property
    def dev_fe_grid(self) -> int:
        return 0

    @property
    def dev_fes_grid(self) -> int:
        return 0

    @property
    def dev_neighbor_list(self) -> int:
        return 0

    @property
    def dev_neighbor_counts(self) -> int:
        return 0

    @property
    def dev_rtable(self) -> int:
        return 0

    @property
    def dev_field_dip1(self) -> int:
        return 0

    @property
    def dev_field_dip2(self) -> int:
        return 0

    @property
    def dev_offset(self) -> int:
        return 0

    @property
    def dev_offsetxyz(self) -> int:
        return 0

    @property
    def dev_scale_coef(self) -> int:
        return 0

    @property
    def dev_khat(self) -> int:
        return 0

    # Operations
    def update_particle_coordinates(self, x_part, y_part, z_part):
        self._ef.updateParticleCoordinates(_to_list(x_part), _to_list(y_part), _to_list(z_part))

    def update_field_coordinates(self, x_field, y_field, z_field):
        self._ef.updateFieldCoordinates(_to_list(x_field), _to_list(y_field), _to_list(z_field))

    def update_dipoles(self, dip_x, dip_y, dip_z):
        self._ef.updateDipoles(_to_list(dip_x), _to_list(dip_y), _to_list(dip_z))

    def update_dipoles_complex(self, dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi):
        self._ef.updateDipolesComplex(
            _to_list(dip_xr), _to_list(dip_xi),
            _to_list(dip_yr), _to_list(dip_yi),
            _to_list(dip_zr), _to_list(dip_zi)
        )

    def update_quadrupoles(self, quad_1, quad_2, quad_3, quad_4, quad_5):
        self._ef.updateQuadrupoles(
            _to_list(quad_1), _to_list(quad_2), _to_list(quad_3),
            _to_list(quad_4), _to_list(quad_5)
        )

    def update_quadrupoles_complex(self, quad_1r, quad_1i, quad_2r, quad_2i, quad_3r, quad_3i, quad_4r, quad_4i, quad_5r, quad_5i):
        self._ef.updateQuadrupolesComplex(
            _to_list(quad_1r), _to_list(quad_1i),
            _to_list(quad_2r), _to_list(quad_2i),
            _to_list(quad_3r), _to_list(quad_3i),
            _to_list(quad_4r), _to_list(quad_4i),
            _to_list(quad_5r), _to_list(quad_5i)
        )

    def set_self_coef(self, self_coef_r, self_coef_i=None):
        num_p = self.num_particles
        if np.iterable(self_coef_r):
            r_list = _to_list(self_coef_r)
        else:
            r_list = [float(self_coef_r)] * num_p

        if self_coef_i is None:
            i_list = [0.0] * len(r_list)
        else:
            if np.iterable(self_coef_i):
                i_list = _to_list(self_coef_i)
            else:
                i_list = [float(self_coef_i)] * num_p

        self._ef.setSelfCoef(r_list, i_list)

    def clear_particles_updated(self):
        self._ef.clearParticlesUpdated()

    def clear_field_points_updated(self):
        self._ef.clearFieldPointsUpdated()

    def clear_dipoles_updated(self):
        self._ef.clearDipolesUpdated()

    def compute_neighbor_list(self, max_neighbors_per_particle=128):
        pass

    def compute_real_space_tables(self):
        pass

    def compute_precalculations(self):
        pass

    def spread_precalcs(self):
        pass

    def contract_precalcs(self):
        pass

    def real_space_precalcs(self):
        pass

    def spread(self, grid_ptr=None):
        pass

    def scale(self, grid_ptr=None):
        pass

    def contract(self, E_point_ptr=None, Es_grid_ptr=None):
        pass

    def real_space(self, E_point_ptr=None):
        pass

    def calculate(self):
        self._ef.calculate()

    # Host copies (numpy array support)
    def get_neighbor_list_host(self):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    def get_real_space_tables_host(self):
        return (np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))

    def get_precalculations_host(self):
        return (np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))

    def get_dipoles_host(self):
        dip_x, dip_y, dip_z = self._ef.getDipolesHost()
        return (np.array(dip_x, dtype=np.float64),
                np.array(dip_y, dtype=np.float64),
                np.array(dip_z, dtype=np.float64))

    def get_dipoles_complex_host(self):
        dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi = self._ef.getDipolesComplexHost()
        return (np.array(dip_xr, dtype=np.float64), np.array(dip_xi, dtype=np.float64),
                np.array(dip_yr, dtype=np.float64), np.array(dip_yi, dtype=np.float64),
                np.array(dip_zr, dtype=np.float64), np.array(dip_zi, dtype=np.float64))

    def get_spread_precalcs_host(self):
        return np.array([], dtype=np.float64), np.array([], dtype=np.int32)

    def get_contract_precalcs_host(self):
        return (np.array([], dtype=np.float64),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int32))

    def get_real_space_precalcs_host(self):
        return 0.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    def get_epoint_host(self, reshape=True):
        raw = np.array(self._ef.getEPointHost(), dtype=np.complex128)
        if not reshape:
            return raw
        
        num_fields = self.num_field_points
        if num_fields == 0:
            num_fields = self.num_particles
            
        num_quads = self.num_quads
        solve_quads = self.solve_quadrupoles

        if solve_quads and num_quads > 0:
            dip_part = raw[:num_fields * 3].reshape((num_fields, 3))
            quad_part = raw[num_fields * 3:].reshape((num_quads, 5))
            return dip_part, quad_part
        else:
            return raw.reshape((num_fields, 3))


class MonodisperseEwaldElectricField:
    """
    Python wrapper for the C++ Ewald_Electric_Field class.
    """
    def __init__(self, box_x, box_y, box_z, errortol, xi, calc_inter_dipole, solve_quadrupoles=False, quad_idxs=None):
        if quad_idxs is None:
            quad_idxs = []
        self._ef = _cuMPM.Monodisperse_Ewald_Electric_Field(
            float(box_x), float(box_y), float(box_z),
            float(errortol), float(xi), bool(calc_inter_dipole),
            bool(solve_quadrupoles), [int(i) for i in quad_idxs]
        )

    @property
    def box_x(self) -> float:
        return self._ef.getBoxX()

    @property
    def box_y(self) -> float:
        return self._ef.getBoxY()

    @property
    def box_z(self) -> float:
        return self._ef.getBoxZ()

    @property
    def errortol(self) -> float:
        return self._ef.getErrortol()

    @property
    def rc(self) -> float:
        return self._ef.getRc()

    @property
    def xi(self) -> float:
        return self._ef.getXi()

    @property
    def calc_inter_dipole(self) -> bool:
        return self._ef.getCalcInterDipole()

    @property
    def self_coef(self) -> float:
        return self._ef.getSelfCoef()

    @property
    def num_particles(self) -> int:
        return self._ef.getNumParticles()

    @property
    def num_field_points(self) -> int:
        return self._ef.getNumFieldPoints()

    @property
    def particles_updated(self) -> bool:
        return self._ef.getParticlesUpdated()

    @property
    def field_points_updated(self) -> bool:
        return self._ef.getFieldPointsUpdated()

    @property
    def dipoles_updated(self) -> bool:
        return self._ef.getDipolesUpdated()

    @property
    def num_spread(self) -> int:
        return self._ef.getNumSpread()

    @property
    def num_contract(self) -> int:
        return self._ef.getNumContract()

    @property
    def num_pairs(self) -> int:
        return self._ef.getNumPairs()

    @property
    def max_neighbors(self) -> int:
        return self._ef.getMaxNeighbors()

    @property
    def table_size(self) -> int:
        return self._ef.getTableSize()

    @property
    def num_offsets(self) -> int:
        return self._ef.getNumOffsets()

    @property
    def solve_quadrupoles(self) -> bool:
        return self._ef.getSolveQuadrupoles()

    @property
    def num_quads(self) -> int:
        return self._ef.getNumQuads()

    @property
    def num_grid(self) -> np.ndarray:
        return np.array(self._ef.getNumGrid(), dtype=np.int32)

    @property
    def grid_spacing(self) -> np.ndarray:
        return np.array(self._ef.getGridSpacing(), dtype=np.float64)

    @property
    def spectral_split(self) -> np.ndarray:
        return np.array(self._ef.getSpectralSplit(), dtype=np.float64)

    # Device pointers
    @property
    def dev_dipoles(self) -> int:
        return self._ef.getDevDipoles()

    @property
    def dev_epoint(self) -> int:
        return self._ef.getDevEPoint()

    @property
    def dev_x_part(self) -> int:
        return self._ef.getDevXPart()

    @property
    def dev_y_part(self) -> int:
        return self._ef.getDevYPart()

    @property
    def dev_z_part(self) -> int:
        return self._ef.getDevZPart()

    @property
    def dev_x_field(self) -> int:
        return self._ef.getDevXField()

    @property
    def dev_y_field(self) -> int:
        return self._ef.getDevYField()

    @property
    def dev_z_field(self) -> int:
        return self._ef.getDevZField()

    @property
    def dev_self_coef_real(self) -> int:
        return self._ef.getDevSelfCoefReal()

    @property
    def dev_self_coef_imag(self) -> int:
        return self._ef.getDevSelfCoefImag()

    @property
    def dev_spread_coef(self) -> int:
        return self._ef.getDevSpreadCoef()

    @property
    def dev_spread_idxs(self) -> int:
        return self._ef.getDevSpreadIdxs()

    @property
    def dev_particle_index(self) -> int:
        return self._ef.getDevParticleIndex()

    @property
    def dev_contract_coef(self) -> int:
        return self._ef.getDevContractCoef()

    @property
    def dev_contract_idxs(self) -> int:
        return self._ef.getDevContractIdxs()

    @property
    def dev_self_perp(self) -> int:
        return self._ef.getDevSelfPerp()

    @property
    def dev_perp(self) -> int:
        return self._ef.getDevPerp()

    @property
    def dev_para(self) -> int:
        return self._ef.getDevPara()

    @property
    def dev_fe_grid(self) -> int:
        return self._ef.getDevFEGrid()

    @property
    def dev_fes_grid(self) -> int:
        return self._ef.getDevFEsGrid()

    @property
    def dev_neighbor_list(self) -> int:
        return self._ef.getDevNeighborList()

    @property
    def dev_neighbor_counts(self) -> int:
        return self._ef.getDevNeighborCounts()

    @property
    def dev_rtable(self) -> int:
        return self._ef.getDevRTable()

    @property
    def dev_field_dip1(self) -> int:
        return self._ef.getDevFieldDip1()

    @property
    def dev_field_dip2(self) -> int:
        return self._ef.getDevFieldDip2()

    @property
    def dev_offset(self) -> int:
        return self._ef.getDevOffset()

    @property
    def dev_offsetxyz(self) -> int:
        return self._ef.getDevOffsetxyz()

    @property
    def dev_scale_coef(self) -> int:
        return self._ef.getDevScaleCoef()

    @property
    def dev_khat(self) -> int:
        return self._ef.getDevKhat()

    # Operations
    def update_particle_coordinates(self, x_part, y_part, z_part):
        self._ef.updateParticleCoordinates(_to_list(x_part), _to_list(y_part), _to_list(z_part))

    def update_field_coordinates(self, x_field, y_field, z_field):
        self._ef.updateFieldCoordinates(_to_list(x_field), _to_list(y_field), _to_list(z_field))

    def update_dipoles(self, dip_x, dip_y, dip_z):
        self._ef.updateDipoles(_to_list(dip_x), _to_list(dip_y), _to_list(dip_z))

    def update_dipoles_complex(self, dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi):
        self._ef.updateDipolesComplex(
            _to_list(dip_xr), _to_list(dip_xi),
            _to_list(dip_yr), _to_list(dip_yi),
            _to_list(dip_zr), _to_list(dip_zi)
        )

    def update_quadrupoles(self, quad_1, quad_2, quad_3, quad_4, quad_5):
        self._ef.updateQuadrupoles(
            _to_list(quad_1), _to_list(quad_2), _to_list(quad_3),
            _to_list(quad_4), _to_list(quad_5)
        )

    def update_quadrupoles_complex(self, quad_1r, quad_1i, quad_2r, quad_2i, quad_3r, quad_3i, quad_4r, quad_4i, quad_5r, quad_5i):
        self._ef.updateQuadrupolesComplex(
            _to_list(quad_1r), _to_list(quad_1i),
            _to_list(quad_2r), _to_list(quad_2i),
            _to_list(quad_3r), _to_list(quad_3i),
            _to_list(quad_4r), _to_list(quad_4i),
            _to_list(quad_5r), _to_list(quad_5i)
        )

    def set_self_coef(self, self_coef_r, self_coef_i=None):
        if self_coef_i is None:
            if np.iterable(self_coef_r):
                r_list = _to_list(self_coef_r)
                i_list = [0.0] * len(r_list)
                self._ef.setSelfCoef(r_list, i_list)
            else:
                self._ef.setSelfCoef(float(self_coef_r), 0.0)
        else:
            if np.iterable(self_coef_r) or np.iterable(self_coef_i):
                self._ef.setSelfCoef(_to_list(self_coef_r), _to_list(self_coef_i))
            else:
                self._ef.setSelfCoef(float(self_coef_r), float(self_coef_i))

    def clear_particles_updated(self):
        self._ef.clearParticlesUpdated()

    def clear_field_points_updated(self):
        self._ef.clearFieldPointsUpdated()

    def clear_dipoles_updated(self):
        self._ef.clearDipolesUpdated()

    def compute_neighbor_list(self, max_neighbors_per_particle=128):
        self._ef.computeNeighborList(int(max_neighbors_per_particle))

    def compute_real_space_tables(self):
        self._ef.computeRealSpaceTables()

    def compute_precalculations(self):
        self._ef.computePrecalculations()

    def spread_precalcs(self):
        self._ef.spreadPrecalcs()

    def contract_precalcs(self):
        self._ef.contractPrecalcs()

    def real_space_precalcs(self):
        self._ef.realSpacePrecalcs()

    def spread(self, grid_ptr=None):
        self._ef.spread(grid_ptr)

    def scale(self, grid_ptr=None):
        self._ef.scale(grid_ptr)

    def contract(self, E_point_ptr=None, Es_grid_ptr=None):
        self._ef.contract(E_point_ptr, Es_grid_ptr)

    def real_space(self, E_point_ptr=None):
        self._ef.realSpace(E_point_ptr)

    def calculate(self):
        self._ef.calculate()

    def clear_epoint(self):
        self._ef.clearEPoint()

    def compute_real_field(self):
        self.clear_epoint()
        self.real_space(None)
        return self.get_epoint_host()

    def compute_inverse_field(self):
        self.calculate()
        total_field = self.get_epoint_host()
        real_field = self.compute_real_field()
        if isinstance(total_field, tuple):
            dip_tot, quad_tot = total_field
            dip_real, quad_real = real_field
            return dip_tot - dip_real, quad_tot - quad_real
        else:
            return total_field - real_field

    # Host copies (numpy array support)
    def get_neighbor_list_host(self):
        list_data, counts_data = self._ef.getNeighborListHost()
        return np.array(list_data, dtype=np.int32), np.array(counts_data, dtype=np.int32)

    def get_real_space_tables_host(self):
        r_table, dip1, dip2 = self._ef.getRealSpaceTablesHost()
        return (np.array(r_table, dtype=np.float64),
                np.array(dip1, dtype=np.float64),
                np.array(dip2, dtype=np.float64))

    def get_precalculations_host(self):
        offset, offsetxyz, scale_coef, khat = self._ef.getPrecalculationsHost()
        return (np.array(offset, dtype=np.int32),
                np.array(offsetxyz, dtype=np.float64),
                np.array(scale_coef, dtype=np.float64),
                np.array(khat, dtype=np.float64))

    def get_dipoles_host(self):
        dip_x, dip_y, dip_z = self._ef.getDipolesHost()
        return (np.array(dip_x, dtype=np.float64),
                np.array(dip_y, dtype=np.float64),
                np.array(dip_z, dtype=np.float64))

    def get_dipoles_complex_host(self):
        dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi = self._ef.getDipolesComplexHost()
        return (np.array(dip_xr, dtype=np.float64), np.array(dip_xi, dtype=np.float64),
                np.array(dip_yr, dtype=np.float64), np.array(dip_yi, dtype=np.float64),
                np.array(dip_zr, dtype=np.float64), np.array(dip_zi, dtype=np.float64))

    def get_spread_precalcs_host(self):
        spread_coef, spread_idxs = self._ef.getSpreadPrecalcsHost()
        return np.array(spread_coef, dtype=np.float64), np.array(spread_idxs, dtype=np.int32)

    def get_contract_precalcs_host(self):
        E_point, particle_index, contract_coef, contract_idxs = self._ef.getContractPrecalcsHost()
        return (np.array(E_point, dtype=np.float64),
                np.array(particle_index, dtype=np.int32),
                np.array(contract_coef, dtype=np.float64),
                np.array(contract_idxs, dtype=np.int32))

    def get_real_space_precalcs_host(self):
        self_perp, perp, para = self._ef.getRealSpacePrecalcsHost()
        return float(self_perp), np.array(perp, dtype=np.float64), np.array(para, dtype=np.float64)

    def get_epoint_host(self, reshape=True):
        raw = np.array(self._ef.getEPointHost(), dtype=np.complex128)
        if not reshape:
            return raw
        
        num_fields = self.num_field_points
        num_quads = self.num_quads
        solve_quads = self.solve_quadrupoles

        if solve_quads and num_quads > 0:
            dip_part = raw[:num_fields * 3].reshape((num_fields, 3))
            quad_part = raw[num_fields * 3:].reshape((num_quads, 5))
            return dip_part, quad_part
        else:
            return raw.reshape((num_fields, 3))


class PolydisperseEwaldElectricField:
    """
    Python wrapper for the C++ Polydisperse_Electric_Field class.
    """
    def __init__(self, box_x, box_y, box_z, errortol, xi, calc_inter_dipole, particle_radii, solve_quadrupoles=False, quad_idxs=None):
        if quad_idxs is None:
            quad_idxs = []
        self._ef = _cuMPM.Polydisperse_Ewald_Electric_Field(
            float(box_x), float(box_y), float(box_z),
            float(errortol), float(xi), bool(calc_inter_dipole),
            _to_list(particle_radii), bool(solve_quadrupoles), [int(i) for i in quad_idxs]
        )

    @property
    def box_x(self) -> float:
        return self._ef.getBoxX()

    @property
    def box_y(self) -> float:
        return self._ef.getBoxY()

    @property
    def box_z(self) -> float:
        return self._ef.getBoxZ()

    @property
    def errortol(self) -> float:
        return self._ef.getErrortol()

    @property
    def rc(self) -> float:
        return self._ef.getRc()

    @property
    def xi(self) -> float:
        return self._ef.getXi()

    @property
    def calc_inter_dipole(self) -> bool:
        return self._ef.getCalcInterDipole()

    @property
    def num_particles(self) -> int:
        return self._ef.getNumParticles()

    @property
    def num_field_points(self) -> int:
        return self._ef.getNumFieldPoints()

    @property
    def particles_updated(self) -> bool:
        return self._ef.getParticlesUpdated()

    @property
    def field_points_updated(self) -> bool:
        return self._ef.getFieldPointsUpdated()

    @property
    def dipoles_updated(self) -> bool:
        return self._ef.getDipolesUpdated()

    @property
    def num_spread(self) -> int:
        return self._ef.getNumSpread()

    @property
    def num_contract(self) -> int:
        return self._ef.getNumContract()

    @property
    def num_pairs(self) -> int:
        return self._ef.getNumPairs()

    @property
    def max_neighbors(self) -> int:
        return self._ef.getMaxNeighbors()

    @property
    def table_size(self) -> int:
        return self._ef.getTableSize()

    @property
    def num_offsets(self) -> int:
        return self._ef.getNumOffsets()

    @property
    def solve_quadrupoles(self) -> bool:
        return self._ef.getSolveQuadrupoles()

    @property
    def num_quads(self) -> int:
        return self._ef.getNumQuads()

    @property
    def eta(self) -> float:
        return self._ef.getEta()

    @property
    def p(self) -> int:
        return self._ef.getP()

    @property
    def num_grid(self) -> np.ndarray:
        return np.array(self._ef.getNumGrid(), dtype=np.int32)

    @property
    def grid_spacing(self) -> np.ndarray:
        return np.array(self._ef.getGridSpacing(), dtype=np.float64)

    @property
    def spectral_split(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    @property
    def self_coef(self) -> float:
        return 0.0

    # Device pointers
    @property
    def dev_dipoles(self) -> int:
        return self._ef.getDevDipoles()

    @property
    def dev_epoint(self) -> int:
        return self._ef.getDevEPoint()

    @property
    def dev_x_part(self) -> int:
        return self._ef.getDevXPart()

    @property
    def dev_y_part(self) -> int:
        return self._ef.getDevYPart()

    @property
    def dev_z_part(self) -> int:
        return self._ef.getDevZPart()

    @property
    def dev_x_field(self) -> int:
        return self._ef.getDevXField()

    @property
    def dev_y_field(self) -> int:
        return self._ef.getDevYField()

    @property
    def dev_z_field(self) -> int:
        return self._ef.getDevZField()

    @property
    def dev_self_coef_real(self) -> int:
        return self._ef.getDevSelfCoefReal()

    @property
    def dev_self_coef_imag(self) -> int:
        return self._ef.getDevSelfCoefImag()

    @property
    def dev_spread_coef(self) -> int:
        return self._ef.getDevSpreadCoef()

    @property
    def dev_spread_idxs(self) -> int:
        return self._ef.getDevSpreadIdxs()

    @property
    def dev_particle_index(self) -> int:
        return self._ef.getDevParticleIndex()

    @property
    def dev_contract_coef(self) -> int:
        return self._ef.getDevContractCoef()

    @property
    def dev_contract_idxs(self) -> int:
        return self._ef.getDevContractIdxs()

    @property
    def dev_perp(self) -> int:
        return self._ef.getDevPerp()

    @property
    def dev_para(self) -> int:
        return self._ef.getDevPara()

    @property
    def dev_fe_grid(self) -> int:
        return self._ef.getDevFEGrid()

    @property
    def dev_fes_grid(self) -> int:
        return self._ef.getDevFEsGrid()

    @property
    def dev_neighbor_list(self) -> int:
        return self._ef.getDevNeighborList()

    @property
    def dev_neighbor_counts(self) -> int:
        return self._ef.getDevNeighborCounts()

    @property
    def dev_rtable(self) -> int:
        return self._ef.getDevRTable()

    @property
    def dev_field_dip1(self) -> int:
        return self._ef.getDevFieldDip1()

    @property
    def dev_field_dip2(self) -> int:
        return self._ef.getDevFieldDip2()

    @property
    def dev_offset(self) -> int:
        return self._ef.getDevOffset()

    @property
    def dev_offsetxyz(self) -> int:
        return self._ef.getDevOffsetxyz()

    @property
    def dev_scale_coef(self) -> int:
        return self._ef.getDevScaleCoef()

    @property
    def dev_self_perp(self) -> int:
        return 0

    @property
    def dev_khat(self) -> int:
        return 0

    @property
    def dev_gpoint(self) -> int:
        return self._ef.getDevGPoint()

    @property
    def dev_quad_idxs(self) -> int:
        return self._ef.getDevQuadIdxs()

    @property
    def dev_quad_map(self) -> int:
        return self._ef.getDevQuadMap()

    @property
    def dev_spread_coef_q(self) -> int:
        return self._ef.getDevSpreadCoefQ()

    @property
    def dev_contract_coef_q(self) -> int:
        return self._ef.getDevContractCoefQ()

    # Operations
    def update_particle_coordinates(self, x_part, y_part, z_part):
        self._ef.updateParticleCoordinates(_to_list(x_part), _to_list(y_part), _to_list(z_part))

    def update_field_coordinates(self, x_field, y_field, z_field):
        self._ef.updateFieldCoordinates(_to_list(x_field), _to_list(y_field), _to_list(z_field))

    def update_dipoles(self, dip_x, dip_y, dip_z):
        self._ef.updateDipoles(_to_list(dip_x), _to_list(dip_y), _to_list(dip_z))

    def update_dipoles_complex(self, dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi):
        self._ef.updateDipolesComplex(
            _to_list(dip_xr), _to_list(dip_xi),
            _to_list(dip_yr), _to_list(dip_yi),
            _to_list(dip_zr), _to_list(dip_zi)
        )

    def update_quadrupoles(self, quad_1, quad_2, quad_3, quad_4, quad_5):
        self._ef.updateQuadrupoles(
            _to_list(quad_1), _to_list(quad_2), _to_list(quad_3),
            _to_list(quad_4), _to_list(quad_5)
        )

    def update_quadrupoles_complex(self, quad_1r, quad_1i, quad_2r, quad_2i, quad_3r, quad_3i, quad_4r, quad_4i, quad_5r, quad_5i):
        self._ef.updateQuadrupolesComplex(
            _to_list(quad_1r), _to_list(quad_1i),
            _to_list(quad_2r), _to_list(quad_2i),
            _to_list(quad_3r), _to_list(quad_3i),
            _to_list(quad_4r), _to_list(quad_4i),
            _to_list(quad_5r), _to_list(quad_5i)
        )

    def set_self_coef(self, self_coef_r, self_coef_i=None):
        if self_coef_i is None:
            if np.iterable(self_coef_r):
                r_list = _to_list(self_coef_r)
                i_list = [0.0] * len(r_list)
                self._ef.setSelfCoef(r_list, i_list)
            else:
                self._ef.setSelfCoef(float(self_coef_r), 0.0)
        else:
            if np.iterable(self_coef_r) or np.iterable(self_coef_i):
                self._ef.setSelfCoef(_to_list(self_coef_r), _to_list(self_coef_i))
            else:
                self._ef.setSelfCoef(float(self_coef_r), float(self_coef_i))

    def clear_particles_updated(self):
        self._ef.clearParticlesUpdated()

    def clear_field_points_updated(self):
        self._ef.clearFieldPointsUpdated()

    def clear_dipoles_updated(self):
        self._ef.clearDipolesUpdated()

    def compute_neighbor_list(self, max_neighbors_per_particle=128):
        self._ef.computeNeighborList(int(max_neighbors_per_particle))

    def compute_real_space_tables(self):
        self._ef.computeRealSpaceTables()

    def compute_precalculations(self):
        self._ef.computePrecalculations()

    def spread_precalcs(self):
        self._ef.spreadPrecalcs()

    def contract_precalcs(self):
        self._ef.contractPrecalcs()

    def real_space_precalcs(self):
        self._ef.realSpacePrecalcs()

    def spread(self, grid_ptr=None):
        self._ef.spread(grid_ptr)

    def scale(self, grid_ptr=None):
        self._ef.scale(grid_ptr)

    def contract(self, E_point_ptr=None, Es_grid_ptr=None):
        self._ef.contract(E_point_ptr, Es_grid_ptr)

    def real_space(self, E_point_ptr=None):
        self._ef.realSpace(E_point_ptr)

    def calculate(self):
        self._ef.calculate()

    def clear_epoint(self):
        self._ef.clearEPoint()

    def compute_real_field(self):
        self.clear_epoint()
        self.real_space(None)
        return self.get_epoint_host()

    def compute_inverse_field(self):
        self.calculate()
        total_field = self.get_epoint_host()
        real_field = self.compute_real_field()
        if isinstance(total_field, tuple):
            dip_tot, quad_tot = total_field
            dip_real, quad_real = real_field
            return dip_tot - dip_real, quad_tot - quad_real
        else:
            return total_field - real_field

    # Host copies (numpy array support)
    def get_neighbor_list_host(self):
        list_data, counts_data = self._ef.getNeighborListHost()
        return np.array(list_data, dtype=np.int32), np.array(counts_data, dtype=np.int32)

    def get_real_space_tables_host(self):
        r_table, dip1, dip2 = self._ef.getRealSpaceTablesHost()
        return (np.array(r_table, dtype=np.float64),
                np.array(dip1, dtype=np.float64),
                np.array(dip2, dtype=np.float64))

    def get_precalculations_host(self):
        offset, offsetxyz, scale_coef = self._ef.getPrecalculationsHost()
        return (np.array(offset, dtype=np.int32),
                np.array(offsetxyz, dtype=np.float64),
                np.array(scale_coef, dtype=np.float64),
                np.array([], dtype=np.float64))

    def get_dipoles_host(self):
        dip_x, dip_y, dip_z = self._ef.getDipolesHost()
        return (np.array(dip_x, dtype=np.float64),
                np.array(dip_y, dtype=np.float64),
                np.array(dip_z, dtype=np.float64))

    def get_dipoles_complex_host(self):
        dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi = self._ef.getDipolesComplexHost()
        return (np.array(dip_xr, dtype=np.float64), np.array(dip_xi, dtype=np.float64),
                np.array(dip_yr, dtype=np.float64), np.array(dip_yi, dtype=np.float64),
                np.array(dip_zr, dtype=np.float64), np.array(dip_zi, dtype=np.float64))

    def get_spread_precalcs_host(self):
        spread_coef, spread_idxs = self._ef.getSpreadPrecalcsHost()
        return np.array(spread_coef, dtype=np.float64), np.array(spread_idxs, dtype=np.int32)

    def get_contract_precalcs_host(self):
        E_point, particle_index, contract_coef, contract_idxs = self._ef.getContractPrecalcsHost()
        return (np.array(E_point, dtype=np.float64),
                np.array(particle_index, dtype=np.int32),
                np.array(contract_coef, dtype=np.float64),
                np.array(contract_idxs, dtype=np.int32))

    def get_real_space_precalcs_host(self):
        # Polydisperse doesn't return self_perp as double, return placeholder
        return 0.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    def get_epoint_host(self, reshape=True):
        raw = np.array(self._ef.getEPointHost(), dtype=np.complex128)
        if not reshape:
            return raw
        
        num_fields = self.num_field_points
        if num_fields == 0:
            num_fields = self.num_particles
            
        num_quads = self.num_quads
        solve_quads = self.solve_quadrupoles

        if solve_quads and num_quads > 0:
            dip_part = raw[:num_fields * 3].reshape((num_fields, 3))
            quad_part = raw[num_fields * 3:].reshape((num_quads, 5))
            return dip_part, quad_part
        else:
            return raw.reshape((num_fields, 3))


EwaldElectricField = MonodisperseEwaldElectricField
PolydisperseElectricField = PolydisperseEwaldElectricField
