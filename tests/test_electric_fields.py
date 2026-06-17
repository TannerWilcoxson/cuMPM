import numpy as np
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

def test_monodisperse_equivalence():
    d = 11.7
    d_opt = 10.0
    eps_m = 2.13
    omega = np.linspace(1000, 7000, 10)

    omega_p1, gamma1, eps_inf1 = 12313, 681, 4
    eps_p1 = drude_dielectric(omega, gamma1, omega_p1, eps_inf1)

    N = 10  # 100 particles
    L = d * N
    A = np.arange(0, L, d)
    y, x = np.meshgrid(A, A)
    pos = np.array([x, y, np.zeros_like(x)]).T.reshape(N**2, 3)
    box = np.array([L, L, 50 * d])

    num_particles = N**2
    eps_p_mixed = np.zeros((len(omega), num_particles), dtype=complex)
    for i in range(num_particles):
        eps_p_mixed[:, i] = eps_p1

    # Monodisperse solver (Option 1)
    mpm_mono = cuMPM.dipole_solver(
        box, eps_p_mixed, radius=d_opt/2, eps_m=eps_m, tol=1e-9, field_type="monodisperse"
    )
    mpm_mono.compute(pos)
    alpha_mono = mpm_mono.get_eff_polarizability()
    dips_mono = mpm_mono.get_dipoles()

    # Polydisperse solver on identical sizes (Option 2)
    mpm_poly_on_mono = cuMPM.dipole_solver(
        box, eps_p_mixed, radius=d_opt/2, eps_m=eps_m, tol=1e-9, field_type="polydisperse"
    )
    mpm_poly_on_mono.compute(pos)
    alpha_poly_on_mono = mpm_poly_on_mono.get_eff_polarizability()
    dips_poly_on_mono = mpm_poly_on_mono.get_dipoles()

    diff_alpha = np.abs(alpha_mono - alpha_poly_on_mono)
    max_diff_alpha = np.max(diff_alpha)

    diff_dips = np.abs(dips_mono - dips_poly_on_mono)
    max_diff_dips = np.max(diff_dips)

    assert max_diff_alpha < 1e-4, f"Option 1 and Option 2 eff polarizabilities do not match! Max diff: {max_diff_alpha}"
    assert max_diff_dips < 1e-4, f"Option 1 and Option 2 particle dipoles do not match! Max diff: {max_diff_dips}"


def test_polydisperse_mixed_sizes():
    d = 11.7
    eps_m = 2.13
    omega = np.linspace(1000, 7000, 10)

    omega_p1, gamma1, eps_inf1 = 12313, 681, 4
    eps_p1 = drude_dielectric(omega, gamma1, omega_p1, eps_inf1)

    N = 10  # 100 particles
    L = d * N
    A = np.arange(0, L, d)
    y, x = np.meshgrid(A, A)
    pos = np.array([x, y, np.zeros_like(x)]).T.reshape(N**2, 3)
    box = np.array([L, L, 50 * d])

    num_particles = N**2
    eps_p_mixed = np.zeros((len(omega), num_particles), dtype=complex)
    for i in range(num_particles):
        eps_p_mixed[:, i] = eps_p1

    # Mixed sizes: alternating 4.0 and 6.0 nm radius
    radii = np.zeros(num_particles)
    for i in range(num_particles):
        radii[i] = 4.0 if i % 2 == 0 else 6.0

    mpm_poly = cuMPM.dipole_solver(
        box, eps_p_mixed, radius=radii, eps_m=eps_m, tol=1e-6, field_type="auto"
    )
    mpm_poly.compute(pos)
    alpha_poly = mpm_poly.get_eff_polarizability()

    assert alpha_poly.shape == (len(omega), 3, 3)
    assert not np.any(np.isnan(alpha_poly))


def test_direct_vs_ewald_large_box():
    """
    Verify that field_type='direct' (open BC) agrees with the Ewald-based
    monodisperse solver when the box is very large relative to the cluster,
    so that periodic images contribute negligibly.
    
    We use a small cluster (5 particles) in a very large box and compare
    the effective polarizabilities from both methods.
    """
    # A small cluster of 5 particles
    d = 5.0          # inter-particle distance (nm)
    radius = 2.0     # particle radius (nm)
    eps_p = 4.0 + 0.2j
    eps_m = 1.0

    # Simple 5-particle cross arrangement
    pos = np.array([
        [0.0,  0.0,  0.0],
        [d,    0.0,  0.0],
        [-d,   0.0,  0.0],
        [0.0,  d,    0.0],
        [0.0, -d,    0.0],
    ])

    # Box >> cluster size: periodic images are ~100 nm away
    L = 500.0
    box = [L, L, L]

    solver_ewald = cuMPM.dipole_solver(box, eps_p, radius=radius, eps_m=eps_m,
                                       tol=1e-5, field_type='monodisperse', quiet=True)
    solver_ewald.compute(pos)
    alpha_ewald = solver_ewald.get_eff_polarizability()

    solver_direct = cuMPM.dipole_solver(box, eps_p, radius=radius, eps_m=eps_m,
                                        tol=1e-5, field_type='direct', quiet=True)
    solver_direct.compute(pos)
    alpha_direct = solver_direct.get_eff_polarizability()

    # In the large-box limit, the two methods should agree within ~0.5%
    np.testing.assert_allclose(alpha_ewald, alpha_direct, rtol=5e-3, atol=1e-5)
