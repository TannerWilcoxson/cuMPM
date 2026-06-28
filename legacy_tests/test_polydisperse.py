import numpy as np
import pytest
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

def test_polydisperse_quadrupole_equivalence():
    """
    Test that the polydisperse quadrupole solver on identical sizes matches
    the monodisperse quadrupole solver.
    """
    d = 30.0
    eps_m = 2.13
    omega = np.linspace(1000, 7000, 5)

    omega_p1, gamma1, eps_inf1 = 12313, 681, 4
    eps_p = drude_dielectric(omega, gamma1, omega_p1, eps_inf1)

    # Let's create a 3x3 grid of particles
    N = 3
    L = d * N
    A = np.arange(0, L, d)
    y, x = np.meshgrid(A, A)
    pos = np.array([x, y, np.zeros_like(x)]).T.reshape(N**2, 3)
    box = np.array([L, L, 50 * d])

    num_particles = N**2
    eps_p_matrix = np.zeros((len(omega), num_particles), dtype=complex)
    for i in range(num_particles):
        eps_p_matrix[:, i] = eps_p

    # 1. Monodisperse solver with quadrupoles
    mpm_mono = cuMPM.dipole_solver(
        box, eps_p_matrix, radius=4.0, eps_m=eps_m, tol=1e-8,
        field_type="monodisperse", quadrupoles=True
    )
    mpm_mono.compute(pos)
    alpha_mono = mpm_mono.get_eff_polarizability()
    dips_mono = mpm_mono.get_dipoles()
    quads_mono = mpm_mono.get_quadrupoles()

    # 2. Polydisperse solver with identical sizes and quadrupoles
    mpm_poly = cuMPM.dipole_solver(
        box, eps_p_matrix, radius=4.0, eps_m=eps_m, tol=1e-8,
        field_type="polydisperse", quadrupoles=True
    )
    mpm_poly.compute(pos)
    alpha_poly = mpm_poly.get_eff_polarizability()
    dips_poly = mpm_poly.get_dipoles()
    quads_poly = mpm_poly.get_quadrupoles()

    # Assert match within tolerance
    np.testing.assert_allclose(alpha_mono, alpha_poly, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dips_mono, dips_poly, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(quads_mono, quads_poly, rtol=1e-5, atol=1e-5)

def test_polydisperse_quadrupole_subset():
    """
    Test that the polydisperse quadrupole solver supports a custom quad_idxs subset
    and matches the monodisperse quadrupole solver.
    """
    d = 10.0
    eps_m = 2.13
    omega = np.linspace(1000, 7000, 3)

    omega_p1, gamma1, eps_inf1 = 12313, 681, 4
    eps_p = drude_dielectric(omega, gamma1, omega_p1, eps_inf1)

    # 4 particles in a line
    pos = np.array([
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0],
        [2*d, 0.0, 0.0],
        [3*d, 0.0, 0.0]
    ])
    num_particles = len(pos)
    box = np.array([20*d, 20*d, 20*d])

    eps_p_matrix = np.zeros((len(omega), num_particles), dtype=complex)
    for i in range(num_particles):
        eps_p_matrix[:, i] = eps_p

    quad_idxs = [0, 2]

    # 1. Monodisperse solver
    mpm_mono = cuMPM.dipole_solver(
        box, eps_p_matrix, radius=4.0, eps_m=eps_m, tol=1e-8,
        field_type="monodisperse", quadrupoles=quad_idxs
    )
    mpm_mono.compute(pos)
    alpha_mono = mpm_mono.get_eff_polarizability()
    dips_mono = mpm_mono.get_dipoles()
    quads_mono = mpm_mono.get_quadrupoles()

    # 2. Polydisperse solver
    mpm_poly = cuMPM.dipole_solver(
        box, eps_p_matrix, radius=4.0, eps_m=eps_m, tol=1e-8,
        field_type="polydisperse", quadrupoles=quad_idxs
    )
    mpm_poly.compute(pos)
    alpha_poly = mpm_poly.get_eff_polarizability()
    dips_poly = mpm_poly.get_dipoles()
    quads_poly = mpm_poly.get_quadrupoles()

    # Check shapes
    assert quads_mono.shape == (len(omega), len(quad_idxs), 3, 5)
    assert quads_poly.shape == (len(omega), len(quad_idxs), 3, 5)

    np.testing.assert_allclose(alpha_mono, alpha_poly, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dips_mono, dips_poly, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(quads_mono, quads_poly, rtol=1e-5, atol=1e-5)

def test_polydisperse_quadrupoles_mixed_sizes():
    """
    Test that the polydisperse quadrupole solver runs without issues when sizes are mixed.
    """
    d = 12.0
    eps_m = 1.5
    eps_p = 5.0 + 0.1j

    pos = np.array([
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0],
        [0.0, d, 0.0]
    ])
    num_particles = len(pos)
    box = np.array([100.0, 100.0, 100.0])

    # Alternating radii
    radii = np.array([3.0, 5.0, 4.0])

    mpm_poly = cuMPM.dipole_solver(
        box, eps_p, radius=radii, eps_m=eps_m, tol=1e-6,
        field_type="polydisperse", quadrupoles=True
    )
    mpm_poly.compute(pos)
    alpha_poly = mpm_poly.get_eff_polarizability()
    dips_poly = mpm_poly.get_dipoles()
    quads_poly = mpm_poly.get_quadrupoles()

    assert alpha_poly.shape == (3, 3)
    assert dips_poly.shape == (num_particles, 3, 3)
    assert quads_poly.shape == (num_particles, 3, 5)
    assert not np.any(np.isnan(alpha_poly))
    assert not np.any(np.isnan(dips_poly))
    assert not np.any(np.isnan(quads_poly))
