import numpy as np
import pytest
from cuMPM import _cuMPM

def test_direct_electric_field_bindings():
    # Instantiation
    radius = [1.0, 1.2, 1.5]
    ef = _cuMPM.Direct_Electric_Field(radius)
    
    # Coordinate updates
    x = [0.0, 1.0, 2.0]
    y = [0.0, 0.0, 0.0]
    z = [0.0, 0.0, 0.0]
    ef.updateParticleCoordinates(x, y, z)
    
    # Initialize dipoles and self coefficients
    ef.updateDipolesComplex([1.0]*3, [0.0]*3, [0.0]*3, [0.0]*3, [0.0]*3, [0.0]*3)
    ef.setSelfCoef([1.0]*3, [0.0]*3)
    
    # Check pointer addresses are returned as ints/longs (uintptr_t)
    dev_dipoles = ef.getDevDipoles()
    dev_epoint = ef.getDevEPoint()
    assert isinstance(dev_dipoles, int)
    assert isinstance(dev_epoint, int)

def test_ewald_electric_field_bindings():
    # Setup small box & errortol to keep grid small
    box_x = 20.0
    box_y = 20.0
    box_z = 20.0
    errortol = 1e-3
    xi = 0.5
    calc_inter_dipole = True
    
    ef = _cuMPM.Monodisperse_Ewald_Electric_Field(box_x, box_y, box_z, errortol, xi, calc_inter_dipole)
    
    # Getters for configuration
    assert ef.getBoxX() == box_x
    assert ef.getBoxY() == box_y
    assert ef.getBoxZ() == box_z
    assert ef.getErrortol() == errortol
    assert ef.getXi() == xi
    assert ef.getCalcInterDipole() == calc_inter_dipole
    
    # Coordinate updates
    x = [0.0, 5.0]
    y = [0.0, 0.0]
    z = [0.0, 0.0]
    ef.updateParticleCoordinates(x, y, z)
    
    assert ef.getNumParticles() == 2
    assert ef.getParticlesUpdated() == True
    
    # Initialize dipoles and self coefficients
    ef.updateDipoles([1.0]*2, [0.0]*2, [0.0]*2)
    ef.setSelfCoef([1.0]*2, [0.0]*2)
    
    # Real space tables and precalculations
    ef.computeNeighborList(32)
    ef.computeRealSpaceTables()
    ef.computePrecalculations()
    
    # Verify arrays/lists are returned correctly
    num_grid = ef.getNumGrid()
    assert len(num_grid) == 3
    assert all(isinstance(val, int) for val in num_grid)
    
    grid_spacing = ef.getGridSpacing()
    assert len(grid_spacing) == 3
    assert all(isinstance(val, float) for val in grid_spacing)
    
    spectral_split = ef.getSpectralSplit()
    assert len(spectral_split) == 3
    assert all(isinstance(val, float) for val in spectral_split)
    
    # Check pointer addresses
    assert isinstance(ef.getDevXPart(), int)
    assert isinstance(ef.getDevFEGrid(), int)
    
    # Pipeline operations
    ef.spreadPrecalcs()
    ef.contractPrecalcs()
    ef.realSpacePrecalcs()
    
    # Default pipeline runs
    ef.spread()
    ef.scale()
    ef.contract()
    ef.realSpace()
    
    # Host copies check
    r_tab, fd1, fd2 = ef.getRealSpaceTablesHost()
    assert isinstance(r_tab, list)
    assert isinstance(fd1, list)
    assert isinstance(fd2, list)

def test_polydisperse_electric_field_bindings():
    # Setup small box & errortol
    box_x = 20.0
    box_y = 20.0
    box_z = 20.0
    errortol = 1e-3
    xi = 0.5
    calc_inter_dipole = True
    radii = [1.0, 2.0]
    
    ef = _cuMPM.Polydisperse_Ewald_Electric_Field(box_x, box_y, box_z, errortol, xi, calc_inter_dipole, radii)
    
    assert ef.getBoxX() == box_x
    assert ef.getXi() == xi
    
    x = [0.0, 5.0]
    y = [0.0, 0.0]
    z = [0.0, 0.0]
    ef.updateParticleCoordinates(x, y, z)
    
    # Initialize dipoles and self-coefficients (crucial for polydisperse kernel)
    ef.updateDipoles([1.0]*2, [0.0]*2, [0.0]*2)
    ef.setSelfCoef([1.0]*2, [0.0]*2)
    
    ef.computeNeighborList(32)
    ef.computeRealSpaceTables()
    ef.computePrecalculations()
    
    # Pipeline operations
    ef.spreadPrecalcs()
    ef.contractPrecalcs()
    ef.realSpacePrecalcs()
    
    ef.spread()
    ef.scale()
    ef.contract()
    ef.realSpace()
    
    assert isinstance(ef.getDevXPart(), int)
    assert isinstance(ef.getDevFEGrid(), int)
