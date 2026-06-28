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
    L = 30.0
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


# def test_quadrupoles():
#     """
#     Verify that the joint dipole-quadrupole solver in C++ matches
#     a Python implementation built from the reference quadrupole_field.py equations.
#     """
#     import sys
#     sys.path.append("/home/tanner/cuMPM")
#     from quadrupole_field import Electric_Field as PyElectric_Field
# 
#     box = np.array([100.0, 100.0, 100.0])
#     xi = 0.4
#     errortol = 1e-6
#     eps_p = 4.5 + 0.2j
#     radius = 1.0
#     eps_m = 1.0
# 
#     pos = np.array([
#         [10.0, 10.0, 10.0],
#         [20.0, 20.0, 20.0]
#     ])
#     num_particles = len(pos)
# 
#     # C++ Solver
#     solver_cpp = cuMPM.dipole_solver(
#         box, eps_p, radius=radius, eps_m=eps_m, xi=xi, tol=1e-8,
#         field_type='monodisperse', quiet=True, quadrupoles=True
#     )
#     solver_cpp.compute(pos)
#     dips_cpp = solver_cpp.get_dipoles()
#     quads_cpp = solver_cpp.get_quadrupoles()
# 
#     # Squeezed C++ solution vectors
#     assert dips_cpp.shape == (num_particles, 3, 3)
#     assert quads_cpp.shape == (num_particles, 3, 5)
# 
#     # Let's build the joint Python system matrix A_py to compare
#     N_dofs = 3 * num_particles + 5 * num_particles
#     A_py = np.zeros((N_dofs, N_dofs), dtype=complex)
#     
#     ef_py = PyElectric_Field(box, xi, errortol, eps_p=eps_p)
#     ef_py.set_points(pos)
#     ef_py.set_dip_pos(pos)
# 
#     for col in range(N_dofs):
#         P_in = np.zeros((num_particles, 3), dtype=complex)
#         Q_in = np.zeros((num_particles, 5), dtype=complex)
#         if col < 3 * num_particles:
#             p_idx = col // 3
#             p_dim = col % 3
#             P_in[p_idx, p_dim] = 1.0
#         else:
#             q_col = col - 3 * num_particles
#             q_idx = q_col // 5
#             q_dim = q_col % 5
#             Q_in[q_idx, q_dim] = 1.0
#         
#         ef_py.set_dipoles(P_in, Q_in)
#         E_out, G_out = ef_py.calculate()
#         
#         col_vec = np.zeros(N_dofs, dtype=complex)
#         col_vec[:3*num_particles] = E_out.ravel()
#         col_vec[3*num_particles:] = G_out.ravel()
#         A_py[:, col] = col_vec
# 
#     for dim in range(3):
#         b = np.zeros(N_dofs, dtype=complex)
#         for p in range(num_particles):
#             b[p * 3 + dim] = 1.0
#             
#         sol_py = np.linalg.solve(A_py, b)
#         
#         dips_py = sol_py[:3*num_particles].reshape(num_particles, 3)
#         quads_py = sol_py[3*num_particles:].reshape(num_particles, 5)
#         
#         np.testing.assert_allclose(dips_cpp[:, dim], dips_py, rtol=1e-5, atol=1e-5)
#         np.testing.assert_allclose(quads_cpp[:, dim], quads_py, rtol=1e-5, atol=1e-5)


# def test_direct_quadrupoles():
#     """
#     Verify that field_type='direct' (open BC) with quadrupoles enabled
#     agrees with the Ewald-based monodisperse solver with quadrupoles enabled
#     in the large-box limit.
#     """
#     box = np.array([100.0, 100.0, 100.0])
#     eps_p = 4.5 + 0.2j
#     radius = 1.0
#     eps_m = 1.0
# 
#     pos = np.array([
#         [0.0,  0.0,  0.0],
#         [3.0,  0.0,  0.0],
#         [-3.0, 0.0,  0.0],
#         [0.0,  3.0,  0.0],
#         [0.0, -3.0,  0.0],
#     ])
#     
#     # 1. Direct Solver
#     solver_direct = cuMPM.dipole_solver(
#         box, eps_p, radius=radius, eps_m=eps_m, tol=1e-5,
#         field_type='direct', quiet=True, quadrupoles=True
#     )
#     solver_direct.compute(pos)
#     alpha_direct = solver_direct.get_eff_polarizability()
#     dips_direct = solver_direct.get_dipoles()
#     quads_direct = solver_direct.get_quadrupoles()
# 
#     # 2. Ewald Solver
#     solver_ewald = cuMPM.dipole_solver(
#         box, eps_p, radius=radius, eps_m=eps_m, xi=0.4, tol=1e-5,
#         field_type='monodisperse', quiet=True, quadrupoles=True
#     )
#     solver_ewald.compute(pos)
#     alpha_ewald = solver_ewald.get_eff_polarizability()
#     dips_ewald = solver_ewald.get_dipoles()
#     quads_ewald = solver_ewald.get_quadrupoles()
# 
#     # Compare
#     np.testing.assert_allclose(alpha_direct, alpha_ewald, rtol=5e-3, atol=1e-5)
#     np.testing.assert_allclose(dips_direct, dips_ewald, rtol=5e-3, atol=1e-5)
#     np.testing.assert_allclose(quads_direct, quads_ewald, rtol=5e-3, atol=1e-5)

