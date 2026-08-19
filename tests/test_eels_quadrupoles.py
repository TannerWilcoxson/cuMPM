import numpy as np
import pytest
import cuMPM

def test_eels_quadrupoles_basic_and_equivalence():
    box = np.array([150.0, 150.0, 150.0])
    eps_p = [4.5 + 0.2j]
    omega = [0.15]
    v = 0.35
    radius = 1.2
    eps_m = 1.0
    tol = 1e-6

    # 3 particles along the z axis
    pos = np.array([
        [0.0, 0.0, -10.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    epos = np.array([12.0, 0.0]) # impact parameter away from particles

    # 1. Direct Solver
    solver_direct = cuMPM.EELS(
        box, eps_p, omega, v, eps_m=eps_m, radius=radius, tol=tol,
        field_type='direct', quiet=True, solve_quadrupoles=True, precision="double"
    )
    solver_direct.compute(epos, pos)
    eels_direct = solver_direct.get_eels()
    dips_direct = solver_direct.get_dipoles()
    quads_direct = solver_direct.get_quadrupoles()

    assert eels_direct.ndim == 0 or eels_direct.shape == (1,)
    assert dips_direct.shape == (3, 3) or dips_direct.shape == (1, 3, 3)
    assert quads_direct.shape == (3, 5) or quads_direct.shape == (1, 3, 5)

    # 2. Ewald Solver
    solver_ewald = cuMPM.EELS(
        box, eps_p, omega, v, eps_m=eps_m, radius=radius, tol=tol,
        field_type='monodisperse', quiet=True, solve_quadrupoles=True, xi=0.35, precision="double"
    )
    solver_ewald.compute(epos, pos)
    eels_ewald = solver_ewald.get_eels()
    dips_ewald = solver_ewald.get_dipoles()
    quads_ewald = solver_ewald.get_quadrupoles()

    # Compare Ewald and Direct solvers in a large box
    np.testing.assert_allclose(eels_direct, eels_ewald, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(dips_direct, dips_ewald, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(quads_direct, quads_ewald, rtol=1e-3, atol=1e-5)


def test_eels_quadrupoles_subset():
    box = np.array([100.0, 100.0, 100.0])
    eps_p = [3.0 + 0.1j]
    omega = [0.2]
    v = 0.4
    radius = 1.0
    tol = 1e-5

    pos = np.array([
        [0.0, 0.0, -5.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0]
    ])
    epos = np.array([6.0, 0.0])

    # Solve quadrupoles only for particles 0 and 2
    quad_idxs = [0, 2]
    solver = cuMPM.EELS(
        box, eps_p, omega, v, radius=radius, tol=tol,
        field_type='direct', quiet=True, solve_quadrupoles=True, quad_idxs=quad_idxs, precision="double"
    )
    solver.compute(epos, pos)
    
    dips = solver.get_dipoles()
    quads = solver.get_quadrupoles()

    assert dips.shape == (3, 3)
    assert quads.shape == (2, 5) # only 2 quadrupole particles
