import sys
sys.path.append("/home/tanner/pyMPM/src")
import pyMPM
import cuMPM
import numpy as np

# Setup parameters
d = 11.7 # nm
N = 1 # small lattice for quick test
L = d * N
A = np.arange(0, L, d)
y, x = np.meshgrid(A, A)
pos = np.array([x, y, np.zeros_like(x)]).T
pos = pos.reshape(N**2, 3)
box = np.array([100*d, 100*d, 100*d])

omega_p = 12313
gamma = 681
eps_inf = 4
omega = np.linspace(1000, 7000, 5) # 5 points
eps_p = pyMPM.drude_dielectric(omega, gamma, omega_p, eps_inf)

d_opt = 10 # nm
eps_m = 2.13

# CPU
mpm_py = pyMPM.MPM(box, eps_p, radius=d_opt/2, eps_m=eps_m, tol=1e-6)
mpm_py.compute(pos)
alpha_py = mpm_py.get_eff_polarizability()

# GPU
mpm_cu = cuMPM.dipole_solver(box, eps_p, radius=d_opt/2, eps_m=eps_m, tol=1e-6)
mpm_cu.compute(pos)
alpha_cu = mpm_cu.get_eff_polarizability()

print("CPU Alpha (first 5):")
print(alpha_py)
print("\nGPU Alpha (first 5):")
print(alpha_cu)
print("\nDifference:")
print(alpha_py - alpha_cu)
