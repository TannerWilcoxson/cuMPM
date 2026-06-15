import numpy as np
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
