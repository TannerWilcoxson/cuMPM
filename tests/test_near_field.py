import numpy as np
import pytest
import cuMPM

try:
    import pyMPM
    HAS_PYMPM = True
except ImportError:
    HAS_PYMPM = False

def test_near_field_basic():
    box = np.array([50.0, 50.0, 50.0])
    E0 = np.array([1.0, 0.0, 0.0])
    radius = 1.0
    
    # 2 particles
    dip_pos = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0]
    ])
    dipoles = np.array([
        [0.1, 0.0, 0.0],
        [-0.1, 0.0, 0.0]
    ])
    
    # field points
    field_points = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0]
    ])
    
    nf = cuMPM.Near_Field(box, E0, radius=radius, dip=dipoles, dip_pos=dip_pos, field_points=field_points)
    intensity = nf.calculate()
    
    assert intensity.shape == (2,)
    assert not np.any(np.isnan(intensity))
    assert np.all(intensity >= 0.0)

@pytest.mark.skipif(not HAS_PYMPM, reason="pyMPM not installed")
def test_near_field_vs_pyMPM():
    import pyMPM.near_field
    
    box = np.array([50.0, 50.0, 50.0])
    E0 = np.array([1.0, 0.0, 0.0])
    radius = 1.0
    
    # 4 particles
    dip_pos = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    dipoles = np.array([
        [0.1 + 0.05j, 0.0, 0.0],
        [-0.1, 0.02j, 0.0],
        [0.0, 0.05, -0.01j],
        [0.01, 0.0, 0.1j]
    ])
    
    # field points
    field_points = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [2.5, 2.5, 2.5]
    ])
    
    # pyMPM (CPU)
    nf_py = pyMPM.near_field.Near_Field(box, E0, radius=radius, dip=dipoles, dip_pos=dip_pos)
    nf_py.set_field_points(field_points)
    intensity_py = nf_py.calculate()
    
    # cuMPM (GPU)
    nf_cu = cuMPM.Near_Field(box, E0, radius=radius, dip=dipoles, dip_pos=dip_pos, field_points=field_points)
    intensity_cu = nf_cu.calculate()
    
    diff = np.abs(intensity_py - intensity_cu)
    max_diff = np.max(diff)
    
    print("intensity_py:", intensity_py)
    print("intensity_cu:", intensity_cu)
    print("max_diff:", max_diff)
    
    assert max_diff < 1e-4, f"cuMPM and pyMPM Near_Field intensities do not match! Max diff: {max_diff}"


def test_near_field_direct_vs_ewald_large_box():
    box = np.array([500.0, 500.0, 500.0])
    E0 = np.array([1.0, 0.0, 0.0])
    radius = 1.0
    
    # 4 particles in a small cluster
    dip_pos = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    dipoles = np.array([
        [0.1 + 0.05j, 0.0, 0.0],
        [-0.1, 0.02j, 0.0],
        [0.0, 0.05, -0.01j],
        [0.01, 0.0, 0.1j]
    ])
    
    # field points
    field_points = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [2.5, 2.5, 2.5]
    ])
    
    nf_ewald = cuMPM.Near_Field(box, E0, radius=radius, dip=dipoles, dip_pos=dip_pos, field_points=field_points, field_type="monodisperse")
    intensity_ewald = nf_ewald.calculate()
    
    nf_direct = cuMPM.Near_Field(box, E0, radius=radius, dip=dipoles, dip_pos=dip_pos, field_points=field_points, field_type="direct")
    intensity_direct = nf_direct.calculate()
    
    diff = np.abs(intensity_ewald - intensity_direct)
    max_diff = np.max(diff)
    
    assert max_diff < 1e-4, f"Direct and Ewald near field intensities do not match in large box! Max diff: {max_diff}"

