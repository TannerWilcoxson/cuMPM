import numpy as np
import pytest

try:
    import pyMPM
    HAS_PYMPM = True
except ImportError:
    HAS_PYMPM = False

@pytest.mark.skipif(not HAS_PYMPM, reason="pyMPM is not installed in the environment")
def test_compare_to_pyMPM():
    import cuMPM
    
    d = 11.7
    N = 10
    L = d * N
    A = np.arange(0, L, d)
    y, x = np.meshgrid(A, A)
    pos = np.array([x, y, np.zeros_like(x)]).T.reshape(N**2, 3)
    box = np.array([L, L, 50 * d])

    omega_p = 12313
    gamma = 681
    eps_inf = 4
    omega = np.linspace(1000, 7000, 10)
    eps_p = pyMPM.drude_dielectric(omega, gamma, omega_p, eps_inf)
    d_opt = 10.0
    eps_m = 2.13

    # CPU Solver (pyMPM)
    mpm_py = pyMPM.MPM(box, eps_p, radius=d_opt/2, eps_m=eps_m)
    mpm_py.compute(pos)
    alpha_py = mpm_py.get_eff_polarizability()

    # GPU Solver (cuMPM)
    mpm_cu = cuMPM.dipole_solver(box, eps_p, radius=d_opt/2, eps_m=eps_m)
    mpm_cu.compute(pos)
    alpha_cu = mpm_cu.get_eff_polarizability()

    diff = np.abs(alpha_py - alpha_cu)
    max_diff = np.max(diff)

    assert max_diff < 1e-4, f"cuMPM and pyMPM polarizabilities do not match! Max diff: {max_diff}"
