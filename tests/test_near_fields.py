import numpy as np
import pytest
import cuMPM

def test_near_field_basic():
    box = np.array([50.0, 50.0, 50.0])
    E0 = np.array([1.0, 0.0, 0.0])
    radius = 1.0
    
    dip_pos = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0]
    ])
    dipoles = np.array([
        [0.1, 0.0, 0.0],
        [-0.1, 0.0, 0.0]
    ])
    
    field_points = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0]
    ])
    
    nf = cuMPM.Near_Field(box, E0, radius=radius, dip=dipoles, dip_pos=dip_pos, field_points=field_points)
    intensity = nf.calculate()
    
    assert intensity.shape == (2,)
    assert not np.any(np.isnan(intensity))
    assert np.all(intensity >= 0.0)

def test_near_field_vs_pyMPM():
    import pyMPM.near_field
    
    box = np.array([50.0, 50.0, 50.0])
    E0 = np.array([1.0, 0.0, 0.0])
    radius = 1.0
    
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
    
    field_points = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [2.5, 2.5, 2.5]
    ])
    
    # pyMPM (CPU)
    import os
    ref_dir = os.path.join(os.path.dirname(__file__), "data")
    ref_path = os.path.join(ref_dir, "ref_near_field.npz")
    if os.path.exists(ref_path):
        ref_data = np.load(ref_path)
        intensity_py = ref_data["intensity_py"]
    else:
        nf_py = pyMPM.near_field.Near_Field(box, E0, radius=radius, dip=dipoles, dip_pos=dip_pos)
        nf_py.set_field_points(field_points)
        intensity_py = nf_py.calculate()
        os.makedirs(ref_dir, exist_ok=True)
        np.savez(ref_path, intensity_py=intensity_py)
    
    # cuMPM (GPU)
    nf_cu = cuMPM.Near_Field(box, E0, radius=radius, dip=dipoles, dip_pos=dip_pos, field_points=field_points)
    intensity_cu = nf_cu.calculate()
    
    np.testing.assert_allclose(intensity_py, intensity_cu, rtol=1e-12, atol=1e-12)

