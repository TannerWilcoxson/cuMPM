import numpy as np
import pytest
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

def test_dipole_solver_monodisperse_vs_pyMPM():
    import pyMPM
    
    d = 12.0
    radius = 4.0
    eps_m = 2.13
    omega = np.linspace(2000, 6000, 3)
    
    omega_p, gamma, eps_inf = 12000, 700, 4.0
    eps_p = drude_dielectric(omega, gamma, omega_p, eps_inf)
    
    pos = np.array([
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0],
        [0.0, d, 0.0],
        [d, d, 0.0]
    ])
    num_particles = len(pos)
    box = np.array([5.0 * d, 5.0 * d, 5.0 * d])
    
    eps_p_matrix = np.zeros((len(omega), num_particles), dtype=complex)
    for i in range(num_particles):
        eps_p_matrix[:, i] = eps_p

    # 1. cuMPM Solver
    mpm_cu = cuMPM.dipole_solver(
        box, eps_p_matrix, radius=radius, eps_m=eps_m, tol=1e-4,
        field_type="monodisperse", solver_type="bicgstab", quiet=True
    )
    mpm_cu.compute(pos)
    alpha_cu = mpm_cu.get_eff_polarizability(physical=False)
    dips_cu = mpm_cu.get_dipoles(physical=False)

    # 2. pyMPM Reference Solver
    import os
    ref_dir = os.path.join(os.path.dirname(__file__), "data")
    ref_path = os.path.join(ref_dir, "ref_dipole_monodisperse.npz")
    if os.path.exists(ref_path):
        ref_data = np.load(ref_path)
        alpha_py = ref_data["alpha_py"]
        dips_py = ref_data["dips_py"]
    else:
        mpm_py = pyMPM.MPM(
            box, eps_p_matrix, radius=radius, eps_m=eps_m, tol=1e-4, quiet=True
        )
        mpm_py.compute(pos)
        alpha_py = mpm_py.get_eff_polarizability()
        dips_py = mpm_py.get_dipoles()
        os.makedirs(ref_dir, exist_ok=True)
        np.savez(ref_path, alpha_py=alpha_py, dips_py=dips_py)

    np.testing.assert_allclose(alpha_cu, alpha_py, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dips_cu, dips_py, rtol=1e-3, atol=1e-3)


def test_dipole_solver_quadrupoles_monodisperse_vs_reference():
    import sys
    from quadrupole_field import Electric_Field as PyElectric_Field

    box = np.array([100.0, 100.0, 100.0])
    xi = 0.4
    errortol = 1e-6
    eps_p = 4.5 + 0.2j
    radius = 1.0
    eps_m = 1.0

    pos = np.array([
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0]
    ])
    num_particles = len(pos)

    # 1. C++ Solver
    solver_cpp = cuMPM.dipole_solver(
        box, eps_p, radius=radius, eps_m=eps_m, xi=xi, tol=1e-8,
        field_type='monodisperse', quiet=True, quadrupoles=True
    )
    solver_cpp.compute(pos)
    dips_cpp = solver_cpp.get_dipoles()
    quads_cpp = solver_cpp.get_quadrupoles()

    assert dips_cpp.shape == (num_particles, 3, 3)
    assert quads_cpp.shape == (num_particles, 3, 5)

    # 2. Reference Python formulation
    import os
    ref_dir = os.path.join(os.path.dirname(__file__), "data")
    ref_path = os.path.join(ref_dir, "ref_quadrupoles_monodisperse.npz")
    if os.path.exists(ref_path):
        ref_data = np.load(ref_path)
        dips_py_all = ref_data["dips_py_all"]
        quads_py_all = ref_data["quads_py_all"]
    else:
        N_dofs = 3 * num_particles + 5 * num_particles
        A_py = np.zeros((N_dofs, N_dofs), dtype=complex)
        
        ef_py = PyElectric_Field(box, xi, errortol, eps_p=eps_p)
        ef_py.set_points(pos)
        ef_py.set_dip_pos(pos)

        for col in range(N_dofs):
            P_in = np.zeros((num_particles, 3), dtype=complex)
            Q_in = np.zeros((num_particles, 5), dtype=complex)
            if col < 3 * num_particles:
                p_idx = col // 3
                p_dim = col % 3
                P_in[p_idx, p_dim] = 1.0
            else:
                q_col = col - 3 * num_particles
                q_idx = q_col // 5
                q_dim = q_col % 5
                Q_in[q_idx, q_dim] = 1.0
            
            ef_py.set_dipoles(P_in, Q_in)
            E_out, G_out = ef_py.calculate()
            
            col_vec = np.zeros(N_dofs, dtype=complex)
            col_vec[:3*num_particles] = E_out.ravel()
            col_vec[3*num_particles:] = G_out.ravel()
            A_py[:, col] = col_vec

        dips_py_all = []
        quads_py_all = []
        for dim in range(3):
            b = np.zeros(N_dofs, dtype=complex)
            for p in range(num_particles):
                b[p * 3 + dim] = 1.0
                
            sol_py = np.linalg.solve(A_py, b)
            dips_py = sol_py[:3*num_particles].reshape(num_particles, 3)
            quads_py = sol_py[3*num_particles:].reshape(num_particles, 5)
            dips_py_all.append(dips_py)
            quads_py_all.append(quads_py)
        
        dips_py_all = np.stack(dips_py_all)
        quads_py_all = np.stack(quads_py_all)
        os.makedirs(ref_dir, exist_ok=True)
        np.savez(ref_path, dips_py_all=dips_py_all, quads_py_all=quads_py_all)

    for dim in range(3):
        np.testing.assert_allclose(dips_cpp[:, dim], dips_py_all[dim], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(quads_cpp[:, dim], quads_py_all[dim], rtol=1e-5, atol=1e-5)

def test_dipole_solver_direct_vs_ewald_large_box():
    box = np.array([120.0, 120.0, 120.0])
    eps_p = 4.5 + 0.2j
    radius = 1.0
    eps_m = 1.0
 
    pos = np.array([
        [0.0,  0.0,  0.0],
        [12.0, 0.0,  0.0],
        [-12.0, 0.0,  0.0],
        [0.0,  12.0,  0.0],
        [0.0, -12.0,  0.0],
    ])
    
    # 1. Direct Solver
    solver_direct = cuMPM.dipole_solver(
        box, eps_p, radius=radius, eps_m=eps_m, tol=1e-5,
        field_type='direct', quiet=True, quadrupoles=True
    )
    solver_direct.compute(pos)
    alpha_direct = solver_direct.get_eff_polarizability()
    dips_direct = solver_direct.get_dipoles()
    quads_direct = solver_direct.get_quadrupoles()
 
    # 2. Ewald Solver
    solver_ewald = cuMPM.dipole_solver(
        box, eps_p, radius=radius, eps_m=eps_m, xi=0.4, tol=1e-5,
        field_type='monodisperse', quiet=True, quadrupoles=True
    )
    solver_ewald.compute(pos)
    alpha_ewald = solver_ewald.get_eff_polarizability()
    dips_ewald = solver_ewald.get_dipoles()
    quads_ewald = solver_ewald.get_quadrupoles()
 
    np.testing.assert_allclose(alpha_direct, alpha_ewald, rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(dips_direct, dips_ewald, rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(quads_direct, quads_ewald, rtol=5e-3, atol=1e-5)


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
    box = np.array([5*d, 5*d, 5*d])

    eps_p_matrix = np.zeros((len(omega), num_particles), dtype=complex)
    for i in range(num_particles):
        eps_p_matrix[:, i] = eps_p

    quad_idxs = [0, 2]

    # 1. Monodisperse solver
    mpm_mono = cuMPM.dipole_solver(
        box, eps_p_matrix, radius=1.0, eps_m=eps_m, tol=1e-6,
        field_type="monodisperse", quadrupoles=quad_idxs
    )
    mpm_mono.compute(pos)
    alpha_mono = mpm_mono.get_eff_polarizability()
    dips_mono = mpm_mono.get_dipoles()
    quads_mono = mpm_mono.get_quadrupoles()

    # 2. Polydisperse solver
    mpm_poly = cuMPM.dipole_solver(
        box, eps_p_matrix, radius=1.0, eps_m=eps_m, tol=1e-6,
        field_type="polydisperse", quadrupoles=quad_idxs
    )
    mpm_poly.compute(pos)
    alpha_poly = mpm_poly.get_eff_polarizability()
    dips_poly = mpm_poly.get_dipoles()
    quads_poly = mpm_poly.get_quadrupoles()

    # Check shapes
    assert quads_mono.shape == (len(omega), len(quad_idxs), 3, 5)
    assert quads_poly.shape == (len(omega), len(quad_idxs), 3, 5)

    np.testing.assert_allclose(alpha_mono, alpha_poly, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(dips_mono, dips_poly, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(quads_mono, quads_poly, rtol=5e-3, atol=5e-3)


def test_polydisperse_solver_direct_vs_ewald_mixed_radii():
    """
    Test the agreement between Direct and Polydisperse Ewald solvers
    for a polydisperse system of particles with different radii in a large box (with quadrupoles).
    """
    d = 12.0
    eps_m = 1.8
    omega = np.linspace(2000, 6000, 3)

    omega_p1, gamma1, eps_inf1 = 12313, 681, 4
    eps_p = drude_dielectric(omega, gamma1, omega_p1, eps_inf1)

    pos = np.array([
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0],
        [-d, 0.0, 0.0],
        [0.0, d, 0.0]
    ])
    num_particles = len(pos)
    box = np.array([120.0, 120.0, 120.0])
    radii = [1.0, 1.2, 0.8, 1.1]

    eps_p_matrix = np.zeros((len(omega), num_particles), dtype=complex)
    for i in range(num_particles):
        eps_p_matrix[:, i] = eps_p

    quad_idxs = [0, 2]

    # 1. Direct Solver
    mpm_direct = cuMPM.dipole_solver(
        box, eps_p_matrix, radius=radii, eps_m=eps_m, tol=1e-6,
        field_type="direct", quadrupoles=quad_idxs
    )
    mpm_direct.compute(pos)
    alpha_direct = mpm_direct.get_eff_polarizability()
    dips_direct = mpm_direct.get_dipoles()
    quads_direct = mpm_direct.get_quadrupoles()

    # 2. Polydisperse Ewald Solver
    mpm_ewald = cuMPM.dipole_solver(
        box, eps_p_matrix, radius=radii, eps_m=eps_m, xi=0.4, tol=1e-6,
        field_type="polydisperse", quadrupoles=quad_idxs
    )
    mpm_ewald.compute(pos)
    alpha_ewald = mpm_ewald.get_eff_polarizability()
    dips_ewald = mpm_ewald.get_dipoles()
    quads_ewald = mpm_ewald.get_quadrupoles()

    # Verify matching (in a large box, Ewald converges to the direct sum)
    np.testing.assert_allclose(alpha_direct, alpha_ewald, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dips_direct, dips_ewald, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(quads_direct, quads_ewald, rtol=1e-3, atol=1e-3)





