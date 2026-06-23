import numpy as np
import pytest
import cuMPM
from scipy.integrate import simpson

def Ebeam_Field_py(pos0, pos, omega, eps_m, v, a=1.0, min_frac=0.2):
    # Pure Python implementation of relativistic point charge field
    import scipy.special
    
    pos0 = pos0 / a
    pos_ = pos / a
    pos_shape = pos.shape
    pos_[..., :2] = pos_[..., :2] - pos0
    xy_ = pos_[..., :2]
    z_ = pos_[..., 2]
    
    omega_scaled = a * omega
    r_ = np.linalg.norm(xy_, axis=-1)[..., np.newaxis]
    r_[r_ < a * min_frac] = a * min_frac
    rhat = xy_ / r_
    
    gamma = (1.0 - eps_m * v**2)**-0.5
    
    prefactor = (4.0 * np.pi * omega_scaled / (v**2 * gamma)) * np.exp(1j * 2.0 * np.pi * omega_scaled * z_ / v)
    xi = 2.0 * np.pi * omega_scaled * r_[..., 0] / (v * gamma)
    
    E = np.zeros(pos_shape, dtype=complex)
    E[..., 2] = prefactor * 1j / gamma * scipy.special.kv(0, xi)
    E[..., :2] = -prefactor[..., np.newaxis] * scipy.special.kv(1, xi)[..., np.newaxis] * rhat
    
    return E

def evaluate_direct_field_py(dipoles, positions, target_points):
    M = len(target_points)
    N = len(positions)
    Eind = np.zeros((M, 3), dtype=complex)
    INV_4PI = 1.0 / (4.0 * np.pi)
    for j in range(M):
        rj = target_points[j]
        E = np.zeros(3, dtype=complex)
        for i in range(N):
            ri = positions[i]
            p = dipoles[i]
            rx = rj[0] - ri[0]
            ry = rj[1] - ri[1]
            rz = rj[2] - ri[2]
            r2 = rx*rx + ry*ry + rz*rz
            if r2 < 1e-18:
                continue
            r = np.sqrt(r2)
            r3 = r2 * r
            inv_r3 = INV_4PI / r3
            p_dot_r = p[0]*rx + p[1]*ry + p[2]*rz
            E += inv_r3 * (p - 3.0 * p_dot_r / r2 * np.array([rx, ry, rz]))
        Eind[j] = E
    return Eind

def test_eels_direct_vs_python():
    # 1. System setup
    box = [100.0, 100.0, 100.0]
    eps_p = [4.0 + 0.1j]
    omega = [0.15] # wavenumber in 1/nm
    v = 0.3
    radius = 2.0
    eps_m = 1.0
    tol = 1e-6
    
    # Dimer aligned along z-axis
    pos = np.array([
        [0.0, 0.0, -3.0],
        [0.0, 0.0, 3.0]
    ])
    
    # Beam impact parameter: 5.0 nm offset from origin in x-direction
    epos = np.array([5.0, 0.0])
    
    # 2. Run new C++ EELS solver (via python interface)
    eels_solver = cuMPM.EELS(box, eps_p, omega, v, eps_m=eps_m, radius=radius, tol=tol, field_type="direct")
    eels_solver.compute(epos, pos)
    
    eels_val_cu = eels_solver.get_eels()
    dips_cu = eels_solver.get_dipoles()
    
    # 3. Reference Python calculation
    # Wavelength scale is radius[0]
    length_scale = radius
    scaled_pos = pos / length_scale
    scaled_epos = epos / length_scale
    scaled_box = np.array(box) / length_scale
    scaled_omega = omega[0] * length_scale
    
    # Calculate incident ebeam field in Python
    E_inc = Ebeam_Field_py(scaled_epos, scaled_pos, scaled_omega, eps_m=1.0, v=v)
    
    # Solve for dipoles using cuMPM.dipole_solver with custom E0
    # Note: E0 takes [K, N, 3] layout, we have 1 polarization
    E0_input = E_inc[np.newaxis, ...]
    solver_ref = cuMPM.dipole_solver(box, eps_p, radius=radius, eps_m=eps_m, tol=tol, field_type="direct", E0=E0_input)
    solver_ref.compute(pos)
    dips_ref = solver_ref.get_dipoles() # shape (1, 1, 2, 3)
    
    # Evaluate induced field along trajectory in Python
    dz = 0.01
    zmin = -scaled_box[2] / 2.0
    zmax = scaled_box[2] / 2.0 - dz
    z_pts = np.arange(zmin, zmax + dz, dz)
    P0 = scaled_pos[0, 2]
    
    Z_pts = z_pts - P0
    Z_pts[Z_pts < zmin] = Z_pts[Z_pts < zmin] + scaled_box[2]
    Z_pts[Z_pts > zmax] = Z_pts[Z_pts > zmax] - scaled_box[2]
    Z_pts = Z_pts + P0
    
    # Target points for electron trajectory
    target_pts = np.zeros((len(z_pts), 3))
    target_pts[:, 0] = scaled_epos[0]
    target_pts[:, 1] = scaled_epos[1]
    target_pts[:, 2] = z_pts
    
    # Calculate direct dipole-dipole field in Python
    dips_flat = dips_ref / length_scale**3 # scale complex dipoles appropriately
    Eind = evaluate_direct_field_py(dips_flat, scaled_pos, target_pts)
    
    # Integrate to get eels probability
    integrand = np.real(-Eind[:, 2] * np.exp(-1j * 2.0 * np.pi * scaled_omega * Z_pts / v))
    integral = simpson(integrand, z_pts)
    eels_val_ref = integral / (2.0 * np.pi**2 * scaled_omega)
    
    print("cuMPM EELS Probability:", eels_val_cu)
    print("Python EELS Probability:", eels_val_ref)
    
    # Compare dipoles
    # C++ returns dipoles normalized to physical units
    # Let's compare relative difference
    np.testing.assert_allclose(dips_cu, dips_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(eels_val_cu, eels_val_ref, rtol=1e-5, atol=1e-5)

def test_eels_ewald_vs_direct_large_box():
    # Verify Ewald boundary conditions match open BCs (direct solver) in the limit of a very large box
    box = [500.0, 500.0, 500.0]
    eps_p = [4.0 + 0.1j]
    omega = [0.15]
    v = 0.3
    radius = 2.0
    eps_m = 1.0
    tol = 1e-5
    
    pos = np.array([
        [0.0, 0.0, -3.0],
        [0.0, 0.0, 3.0]
    ])
    epos = np.array([5.0, 0.0])
    
    # Direct
    eels_direct = cuMPM.EELS(box, eps_p, omega, v, eps_m=eps_m, radius=radius, tol=tol, field_type="direct")
    eels_direct.compute(epos, pos)
    val_direct = eels_direct.get_eels()
    
    # Ewald
    eels_ewald = cuMPM.EELS(box, eps_p, omega, v, eps_m=eps_m, radius=radius, tol=tol, field_type="monodisperse")
    eels_ewald.compute(epos, pos)
    val_ewald = eels_ewald.get_eels()
    
    print("Direct EELS:", val_direct)
    print("Ewald EELS:", val_ewald)
    
    # Match within 1% in the large box limit
    np.testing.assert_allclose(val_ewald, val_direct, rtol=1e-2, atol=1e-5)

def test_eels_vs_python_module():
    import sys
    import os
    
    # Safely import the reference eels package installed in the environment
    original_path = list(sys.path)
    try:
        # Exclude the local workspace from sys.path to prevent it shadowing the reference eels package
        sys.path = [p for p in sys.path if p and os.path.abspath(p) != '/home/tanner/cuMPM']
        import eels as ref_eels_pkg
    finally:
        sys.path = original_path

    # Extract dimer simulation parameters from dimer.py
    d = 25.0 + 2.6
    d_opt = 25.0
    R = 0.5 * d_opt
    
    box = np.array([10.0, 10.0, 10.0]) * d
    points = np.array([[-0.5, 0.0, 0.0],
                       [0.5, 0.0, 0.0]]) * d
    e_pos = np.array([[0.0, 0.0],
                      [1.0, 0.0]]) * d
    
    omega_p = 11886.0
    gamma = 845.0
    eps_inf = 2.25
    eps_m = 1.0
    
    omega = np.linspace(2000.0, 9000.0, 112)
    eps_p = ref_eels_pkg.drude_dielectric(omega, gamma, omega_p, eps_inf)
    omega_nm = omega / 1e7
    v = 0.45
    
    # 1. Run reference Python EELS calculation
    ref_solver = ref_eels_pkg.EELS(
        box=box,
        eps_p=eps_p,
        omega=omega_nm,
        v=v,
        eps_m=eps_m,
        radius=R,
        xi=0.5,
        ef_method="direct",
        cutoff=500.0,
        split_dist=0.0
    )
    ref_eels, ref_dips, ref_pos, ref_E = ref_solver.compute(e_pos, points)
    
    # Stack list of frame dipoles into a single array of shape (num_frames, num_wavelengths, num_particles, 3)
    ref_dips_stacked = np.stack(ref_dips)
    
    # 2. Run C++ cuMPM EELS calculation
    cu_solver = cuMPM.EELS(
        box=box,
        eps_p=eps_p,
        omega=omega_nm,
        v=v,
        eps_m=eps_m,
        radius=R,
        xi=0.5,
        tol=1e-3,
        field_type="direct",
        solver_type="bicgstab"
    )
    # Tiling positions to match shape (num_frames, num_particles, 3)
    positions_cu = np.tile(points, (len(e_pos), 1, 1))
    cu_solver.compute(e_pos, positions_cu)
    
    cu_eels = cu_solver.get_eels()
    cu_dips = cu_solver.get_dipoles()
    
    print("Reference EELS shape:", ref_eels.shape)
    print("cuMPM EELS shape:", cu_eels.shape)
    
    # 3. Assert close agreement
    np.testing.assert_allclose(cu_eels, ref_eels, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cu_dips, ref_dips_stacked, rtol=1e-5, atol=1e-5)

