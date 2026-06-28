import numpy as np
import pytest
from cuMPM.dev import DirectElectricField, EwaldElectricField, PolydisperseElectricField

def test_direct_electric_field_wrapper():
    # Instantiation
    radius = np.array([1.0, 1.2, 1.5])
    ef = DirectElectricField(radius)
    
    # Coordinate updates using NumPy arrays
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, 0.0])
    ef.update_particle_coordinates(x, y, z)
    
    # Update field coordinates using lists
    ef.update_field_coordinates([0.5, 1.5], [0.0, 0.0], [0.0, 0.0])
    
    # Initialize dipoles and self coefficients
    ef.update_dipoles_complex(
        np.ones(3), np.zeros(3),
        np.zeros(3), np.zeros(3),
        np.zeros(3), np.zeros(3)
    )
    ef.set_self_coef(np.ones(3), np.zeros(3))
    
    # Check properties
    assert isinstance(ef.dev_dipoles, int)
    assert isinstance(ef.dev_epoint, int)
    
    # Perform calculation
    ef.calculate()


def test_ewald_electric_field_wrapper():
    box_x = 20.0
    box_y = 20.0
    box_z = 20.0
    errortol = 1e-3
    xi = 0.5
    calc_inter_dipole = True
    
    ef = EwaldElectricField(box_x, box_y, box_z, errortol, xi, calc_inter_dipole)
    
    # Check configurations and state getters
    assert ef.box_x == box_x
    assert ef.box_y == box_y
    assert ef.box_z == box_z
    assert ef.errortol == errortol
    assert ef.xi == xi
    assert ef.calc_inter_dipole == calc_inter_dipole
    
    # Grid properties
    num_grid = ef.num_grid
    assert isinstance(num_grid, np.ndarray)
    assert num_grid.shape == (3,)
    assert num_grid.dtype == np.int32
    
    grid_spacing = ef.grid_spacing
    assert isinstance(grid_spacing, np.ndarray)
    assert grid_spacing.shape == (3,)
    
    spectral_split = ef.spectral_split
    assert isinstance(spectral_split, np.ndarray)
    assert spectral_split.shape == (3,)

    # Coordinate updates
    x = np.array([0.0, 5.0])
    y = np.array([0.0, 0.0])
    z = np.array([0.0, 0.0])
    ef.update_particle_coordinates(x, y, z)
    
    assert ef.num_particles == 2
    assert ef.particles_updated is True
    
    # Initialize dipoles and self-coefficients
    ef.update_dipoles(np.ones(2), np.zeros(2), np.zeros(2))
    ef.set_self_coef(np.ones(2), np.zeros(2))
    
    # Real space tables and precalculations
    ef.compute_neighbor_list(32)
    ef.compute_real_space_tables()
    ef.compute_precalculations()
    
    # Pipeline operations
    ef.spread_precalcs()
    ef.contract_precalcs()
    ef.real_space_precalcs()
    
    ef.spread()
    ef.scale()
    ef.contract()
    ef.real_space()
    
    # Verify properties (device pointers)
    assert isinstance(ef.dev_x_part, int)
    assert isinstance(ef.dev_fe_grid, int)
    
    # Host copies check
    r_tab, fd1, fd2 = ef.get_real_space_tables_host()
    assert isinstance(r_tab, np.ndarray)
    assert isinstance(fd1, np.ndarray)
    assert isinstance(fd2, np.ndarray)
    
    # epoint host copy
    epoint = ef.get_epoint_host(reshape=True)
    assert isinstance(epoint, np.ndarray)
    assert epoint.shape == (ef.num_field_points, 3)
    assert epoint.dtype == np.complex128


def test_polydisperse_electric_field_wrapper():
    box_x = 20.0
    box_y = 20.0
    box_z = 20.0
    errortol = 1e-3
    xi = 0.5
    calc_inter_dipole = True
    radii = np.array([1.0, 2.0])
    
    ef = PolydisperseElectricField(box_x, box_y, box_z, errortol, xi, calc_inter_dipole, radii)
    
    assert ef.box_x == box_x
    assert ef.xi == xi
    
    x = np.array([0.0, 5.0])
    y = np.array([0.0, 0.0])
    z = np.array([0.0, 0.0])
    ef.update_particle_coordinates(x, y, z)
    
    # Initialize dipoles and self coefficients
    ef.update_dipoles(np.ones(2), np.zeros(2), np.zeros(2))
    ef.set_self_coef(np.ones(2), np.zeros(2))
    
    ef.compute_neighbor_list(32)
    ef.compute_real_space_tables()
    ef.compute_precalculations()
    
    ef.spread_precalcs()
    ef.contract_precalcs()
    ef.real_space_precalcs()
    
    ef.spread()
    ef.scale()
    ef.contract()
    ef.real_space()
    
    assert isinstance(ef.dev_x_part, int)
    assert isinstance(ef.dev_fe_grid, int)
    assert ef.eta > 0.0
    assert ef.p > 0
