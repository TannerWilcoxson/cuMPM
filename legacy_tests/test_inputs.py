import pytest
import numpy as np
import cuMPM

def test_invalid_box_size():
    box = [10.0, 10.0]
    eps_p = 4.0
    with pytest.raises(ValueError, match="box must have length 3"):
        cuMPM.dipole_solver(box, eps_p)

def test_invalid_field_type():
    box = [10.0, 10.0, 10.0]
    eps_p = 4.0
    with pytest.raises(ValueError, match="field_type must be 'auto'"):
        cuMPM.dipole_solver(box, eps_p, field_type="invalid_type")

def test_coordinate_length_mismatch():
    box = [10.0, 10.0, 10.0]
    eps_p = 4.0
    mpm = cuMPM.dipole_solver(box, eps_p)
    # Different coordinate lengths
    x = [0.0, 1.0]
    y = [0.0]
    z = [0.0, 0.0]
    with pytest.raises(Exception):
        # Passing mismatching coordinates
        mpm._solver.compute(x, y, z)

def test_radii_count_mismatch():
    box = [10.0, 10.0, 10.0]
    eps_p = 4.0
    radii = [1.0, 1.0, 1.0]
    mpm = cuMPM.dipole_solver(box, eps_p, radius=radii)
    pos = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
    with pytest.raises(Exception, match="inconsistent with the number of radii"):
        mpm.compute(pos)

def test_monodisperse_with_mixed_radii():
    box = [10.0, 10.0, 10.0]
    eps_p = 4.0
    radii = [1.0, 2.0]
    mpm = cuMPM.dipole_solver(box, eps_p, radius=radii, field_type="monodisperse")
    pos = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
    with pytest.raises(Exception, match="field_type was set to 'monodisperse' but radii are not identical"):
        mpm.compute(pos)
