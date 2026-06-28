import numpy as np
import pytest
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

def test_gmres_vs_bicgstab():
    # Parameters
    d = 11.7
    d_opt = 10.0
    eps_m = 2.13
    omega = np.linspace(1000, 7000, 80)

    omega_p1, gamma1, eps_inf1 = 12313, 681, 4
    eps_p1 = drude_dielectric(omega, gamma1, omega_p1, eps_inf1)

    omega_p2, gamma2, eps_inf2 = 8000, 400, 2
    eps_p2 = drude_dielectric(omega, gamma2, omega_p2, eps_inf2)

    N = 10  # 100 particles
    L = d * N
    A = np.arange(0, L, d)
    y, x = np.meshgrid(A, A)
    pos = np.array([x, y, np.zeros_like(x)]).T.reshape(N**2, 3)
    box = np.array([L, L, 50 * d])

    num_particles = N**2
    eps_p_mixed = np.zeros((len(omega), num_particles), dtype=complex)

    for iy in range(N):
        for ix in range(N):
            idx = iy * N + ix
            if (ix + iy) % 2 == 0:
                eps_p_mixed[:, idx] = eps_p1
            else:
                eps_p_mixed[:, idx] = eps_p2

    mpm_gmres = cuMPM.dipole_solver(box, eps_p_mixed, radius=d_opt/2, eps_m=eps_m, tol=1e-6, solver_type="gmres")
    mpm_gmres.compute(pos)
    alpha_gmres = mpm_gmres.get_eff_polarizability()

    mpm_bicg = cuMPM.dipole_solver(box, eps_p_mixed, radius=d_opt/2, eps_m=eps_m, tol=1e-6, solver_type="bicgstab")
    mpm_bicg.compute(pos)
    alpha_bicg = mpm_bicg.get_eff_polarizability()

    diff = np.abs(alpha_gmres - alpha_bicg)
    max_diff = np.max(diff)
    
    assert max_diff < 1e-4, f"GMRES and BiCGSTAB polarizabilities do not match! Max diff: {max_diff}"


def test_custom_E0():
    # Parameters
    box = [50.0, 50.0, 50.0]
    eps_p = 4.0 + 0.1j
    radius = 1.0
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])

    # 1. Standard basis solver
    solver_std = cuMPM.dipole_solver(box, eps_p, radius=radius, E0=None)
    solver_std.compute(pos)
    alpha_std = solver_std.get_eff_polarizability() # Shape: (3, 3)
    dips_std = solver_std.get_dipoles() # Shape: (2, 3, 3) (num_particles, K, 3)

    # 2. Custom 1D vector E0
    E0_single = [1.0, 0.0, 0.0]
    solver_single = cuMPM.dipole_solver(box, eps_p, radius=radius, E0=E0_single)
    solver_single.compute(pos)
    alpha_single = solver_single.get_eff_polarizability() # Shape: (3,)
    dips_single = solver_single.get_dipoles() # Shape: (2, 3)

    # 3. Custom 2D list of vectors E0
    E0_double = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    solver_double = cuMPM.dipole_solver(box, eps_p, radius=radius, E0=E0_double)
    solver_double.compute(pos)
    alpha_double = solver_double.get_eff_polarizability() # Shape: (2, 3)
    dips_double = solver_double.get_dipoles() # Shape: (2, 2, 3)

    # Assert shapes
    assert alpha_single.shape == (3,)
    assert alpha_double.shape == (2, 3)

    # Assert values match
    np.testing.assert_allclose(alpha_single, alpha_std[0], rtol=1e-5)
    np.testing.assert_allclose(alpha_double[0], alpha_std[0], rtol=1e-5)
    np.testing.assert_allclose(alpha_double[1], alpha_std[1], rtol=1e-5)


def test_per_particle_E0():
    # Parameters
    box = [50.0, 50.0, 50.0]
    eps_p = 4.0 + 0.1j
    radius = 1.0
    pos = np.array([[0.0, 0.0, 0.0], [5000.0, 0.0, 0.0]]) # extremely far apart to ignore interaction

    # 1. Uniform field test for comparison (using direct solver)
    solver_p0 = cuMPM.dipole_solver(box, eps_p, radius=radius, E0=[1.0, 0.0, 0.0], field_type="direct", tol=1e-10)
    solver_p0.compute(pos)
    dips_p0 = solver_p0.get_dipoles() # Shape: (2, 3)

    solver_p1 = cuMPM.dipole_solver(box, eps_p, radius=radius, E0=[0.0, 2.0, 0.0], field_type="direct", tol=1e-10)
    solver_p1.compute(pos)
    dips_p1 = solver_p1.get_dipoles() # Shape: (2, 3)

    # 2. Per-particle field passed as 3D array (shape [1, 2, 3])
    E0_3d = np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    solver_3d = cuMPM.dipole_solver(box, eps_p, radius=radius, E0=E0_3d, field_type="direct", tol=1e-10)
    solver_3d.compute(pos)
    dips_3d = solver_3d.get_dipoles() # Shape: (2, 3)

    # 3. Per-particle field passed as 1D array (shape [6])
    E0_1d = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    solver_1d = cuMPM.dipole_solver(box, eps_p, radius=radius, E0=E0_1d, field_type="direct", tol=1e-10)
    solver_1d.compute(pos)
    dips_1d = solver_1d.get_dipoles() # Shape: (2, 3)

    # Assert shape
    assert dips_3d.shape == (2, 3)
    assert dips_1d.shape == (2, 3)

    # Assert that particle 0's dipole in the per-particle case matches uniform solver_p0
    np.testing.assert_allclose(dips_3d[0], dips_p0[0], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dips_1d[0], dips_p0[0], rtol=1e-5, atol=1e-8)

    # Assert that particle 1's dipole in the per-particle case matches uniform solver_p1
    np.testing.assert_allclose(dips_3d[1], dips_p1[1], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dips_1d[1], dips_p1[1], rtol=1e-5, atol=1e-8)






