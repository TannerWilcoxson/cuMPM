"""
Direct matvec comparison: Apply the same dipoles to both Direct and Ewald
electric field classes and compare the resulting E_point output.
This isolates the field computation from the iterative solver.
"""
import numpy as np
from cuMPM.dev.electric_field import DirectElectricField, MonodisperseEwaldElectricField

np.set_printoptions(precision=10, linewidth=200, suppress=False)

# === Setup ===
N = 3
box = 120.0
xi = 0.4
errortol = 1e-6

pos_x = [0.0, 12.0, 0.0]
pos_y = [0.0, 0.0, 12.0]
pos_z = [0.0, 0.0, 0.0]

radius = [1.0, 1.0, 1.0]

# Self-coefficient matching what dipole_solver.cu computes
eps_p = 4.5 + 0.2j
eps_m = 1.0
diff = 1.0 - eps_p
sc = -3.0 / (4 * np.pi * diff)  # for unit radius

print(f"Self coefficient: {sc}")
print(f"  real: {sc.real}")
print(f"  imag: {sc.imag}")

# === Test 1: Dipole-only field ===
print("\n" + "="*70)
print("TEST 1: Pure dipole field (no self-coef, no quadrupoles)")
print("  Set px=[1,0,0] on particle 0 only, check E at all particles")
print("="*70)

# Direct solver
ef_direct = DirectElectricField(radius=radius, solve_quadrupoles=False)
ef_direct.update_particle_coordinates(pos_x, pos_y, pos_z)
# Zero self-coefs so we only see the inter-particle field
ef_direct.set_self_coef([0.0]*N, [0.0]*N)

# Ewald solver
ef_ewald = MonodisperseEwaldElectricField(box, box, box, errortol, xi, True, solve_quadrupoles=False)
ef_ewald.update_particle_coordinates(pos_x, pos_y, pos_z)
ef_ewald.set_self_coef([0.0]*N, [0.0]*N)

# Set dipoles: only particle 0 has px=1
dip_xr = [1.0, 0.0, 0.0]
dip_xi = [0.0, 0.0, 0.0]
dip_yr = [0.0, 0.0, 0.0]
dip_yi = [0.0, 0.0, 0.0]
dip_zr = [0.0, 0.0, 0.0]
dip_zi = [0.0, 0.0, 0.0]

ef_direct.update_dipoles_complex(dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi)
ef_ewald.update_dipoles_complex(dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi)

ef_direct.calculate()
ef_ewald.calculate()

E_direct = ef_direct.get_epoint_host()
E_ewald = ef_ewald.get_epoint_host()

print("E_direct (N,3):\n", E_direct)
print("E_ewald (N,3):\n", E_ewald)
print("Difference:\n", E_direct - E_ewald)

# Analytical check: dipole p=(1,0,0) at origin, field at (12,0,0):
# r_hat = (1,0,0), r = 12
# E = (3*(p·r̂)*r̂ - p) / (4π r³) = (3*1*1 - 1, 0, 0) / (4π * 12³) = (2, 0, 0) / (4π * 1728)
# E_x = 2 / (4π * 1728) = 9.2106e-05
E_analytical = 2.0 / (4 * np.pi * 12**3)
print(f"\nAnalytical E_x at (12,0,0) from dipole at origin: {E_analytical:.10e}")
print(f"Direct gives E_x at particle 1: {E_direct[1,0]}")
print(f"Ewald gives  E_x at particle 1: {E_ewald[1,0]}")

# === Test 2: With quadrupoles ===
print("\n" + "="*70)
print("TEST 2: Quadrupole field (no self-coef)")
print("  Set Q_xx=1 on particle 0 only, check E at all particles")
print("="*70)

ef_direct_q = DirectElectricField(radius=radius, solve_quadrupoles=True)
ef_direct_q.update_particle_coordinates(pos_x, pos_y, pos_z)
ef_direct_q.set_self_coef([0.0]*N, [0.0]*N)

ef_ewald_q = MonodisperseEwaldElectricField(box, box, box, errortol, xi, True, solve_quadrupoles=True)
ef_ewald_q.update_particle_coordinates(pos_x, pos_y, pos_z)
ef_ewald_q.set_self_coef([0.0]*N, [0.0]*N)

# All dipoles zero, only Q_xx = 1 on particle 0
dip_xr = [0.0, 0.0, 0.0]
dip_xi = [0.0, 0.0, 0.0]
dip_yr = [0.0, 0.0, 0.0]
dip_yi = [0.0, 0.0, 0.0]
dip_zr = [0.0, 0.0, 0.0]
dip_zi = [0.0, 0.0, 0.0]

ef_direct_q.update_dipoles_complex(dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi)
ef_ewald_q.update_dipoles_complex(dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi)

# Set Q_xx = 1 on particle 0 (q0=Q_xx, q1=Q_xy, q2=Q_xz, q3=Q_yy, q4=Q_yz)
q1r = [1.0, 0.0, 0.0]; q1i = [0.0, 0.0, 0.0]  # Q_xx
q2r = [0.0, 0.0, 0.0]; q2i = [0.0, 0.0, 0.0]  # Q_xy
q3r = [0.0, 0.0, 0.0]; q3i = [0.0, 0.0, 0.0]  # Q_xz
q4r = [0.0, 0.0, 0.0]; q4i = [0.0, 0.0, 0.0]  # Q_yy
q5r = [0.0, 0.0, 0.0]; q5i = [0.0, 0.0, 0.0]  # Q_yz

ef_direct_q.update_quadrupoles_complex(q1r, q1i, q2r, q2i, q3r, q3i, q4r, q4i, q5r, q5i)
ef_ewald_q.update_quadrupoles_complex(q1r, q1i, q2r, q2i, q3r, q3i, q4r, q4i, q5r, q5i)

ef_direct_q.calculate()
ef_ewald_q.calculate()

E_direct_q, G_direct_q = ef_direct_q.get_epoint_host()
E_ewald_q, G_ewald_q = ef_ewald_q.get_epoint_host()

print("E_direct (N,3):\n", E_direct_q)
print("E_ewald (N,3):\n", E_ewald_q)
print("E difference:\n", E_direct_q - E_ewald_q)

print("\nG_direct (N,5):\n", G_direct_q)
print("G_ewald (N,5):\n", G_ewald_q)
print("G difference:\n", G_direct_q - G_ewald_q)

# === Test 3: Dipole gradient ===
print("\n" + "="*70)
print("TEST 3: Dipole gradient (px=1 on particle 0 → gradient at all quads)")
print("="*70)

ef_direct_q2 = DirectElectricField(radius=radius, solve_quadrupoles=True)
ef_direct_q2.update_particle_coordinates(pos_x, pos_y, pos_z)
ef_direct_q2.set_self_coef([0.0]*N, [0.0]*N)

ef_ewald_q2 = MonodisperseEwaldElectricField(box, box, box, errortol, xi, True, solve_quadrupoles=True)
ef_ewald_q2.update_particle_coordinates(pos_x, pos_y, pos_z)
ef_ewald_q2.set_self_coef([0.0]*N, [0.0]*N)

# px=1 on particle 0, no quads
dip_xr = [1.0, 0.0, 0.0]
dip_xi = [0.0, 0.0, 0.0]
dip_yr = [0.0, 0.0, 0.0]
dip_yi = [0.0, 0.0, 0.0]
dip_zr = [0.0, 0.0, 0.0]
dip_zi = [0.0, 0.0, 0.0]

ef_direct_q2.update_dipoles_complex(dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi)
ef_ewald_q2.update_dipoles_complex(dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi)

# Zero quadrupoles
q0 = [0.0]*N
ef_direct_q2.update_quadrupoles_complex(q0, q0, q0, q0, q0, q0, q0, q0, q0, q0)
ef_ewald_q2.update_quadrupoles_complex(q0, q0, q0, q0, q0, q0, q0, q0, q0, q0)

ef_direct_q2.calculate()
ef_ewald_q2.calculate()

E_direct2, G_direct2 = ef_direct_q2.get_epoint_host()
E_ewald2, G_ewald2 = ef_ewald_q2.get_epoint_host()

print("E_direct (N,3):\n", E_direct2)
print("E_ewald (N,3):\n", E_ewald2)
print("E difference:\n", E_direct2 - E_ewald2)

print("\nG_direct (N,5) [dipole → gradient]:\n", G_direct2)
print("G_ewald (N,5) [dipole → gradient]:\n", G_ewald2)
print("G difference:\n", G_direct2 - G_ewald2)

# Analytical: For px=1 at origin, gradient at (12,0,0):
# The gradient tensor G_ab = dE_a/dr_b evaluated in the traceless-symmetric 5-component form
# G_0 = dE_x/dx - dE_z/dz (xx - zz)
# For r_hat = (1,0,0), r=12:
# dE/dr = -3/(4pi r^4) * [5*(p·r̂)*r̂r̂ - (p*r̂ + r̂*p) - (p·r̂)I]
# With p·r̂ = 1:
# dE_x/dx = -3/(4pi * 12^4) * (5*1*1 - 2 - 1) = -3/(4pi * 20736) * 2 = -6/(4pi * 20736)
# dE_z/dz = -3/(4pi * 12^4) * (0 - 0 - 1) = +3/(4pi * 20736) 
# G_0 = dE_x/dx - dE_z/dz
r = 12.0
dEdx_xx = -3.0 / (4 * np.pi * r**4) * (5.0 * 1.0 - 2.0 - 1.0)  # = -6/(4pi*r^4)
dEdz_zz = -3.0 / (4 * np.pi * r**4) * (0.0 - 0.0 - 1.0)        # = +3/(4pi*r^4)
G0_analytical = dEdx_xx - dEdz_zz
print(f"\nAnalytical G_0 at (12,0,0): {G0_analytical:.10e}")
print(f"Direct G_0 at particle 1: {G_direct2[1, 0]}")
print(f"Ewald  G_0 at particle 1: {G_ewald2[1, 0]}")

# === Test 4: Self-coefficient test ===
print("\n" + "="*70)
print("TEST 4: Self-coefficient test (px=1, all particles, with self-coef)")
print("="*70)

ef_direct_s = DirectElectricField(radius=radius, solve_quadrupoles=False)
ef_direct_s.update_particle_coordinates(pos_x, pos_y, pos_z)
ef_direct_s.set_self_coef([sc.real]*N, [sc.imag]*N)

ef_ewald_s = MonodisperseEwaldElectricField(box, box, box, errortol, xi, True, solve_quadrupoles=False)
ef_ewald_s.update_particle_coordinates(pos_x, pos_y, pos_z)
ef_ewald_s.set_self_coef([sc.real]*N, [sc.imag]*N)

# px=1 on all particles
dip_xr = [1.0]*N
dip_xi = [0.0]*N
dip_yr = [0.0]*N
dip_yi = [0.0]*N
dip_zr = [0.0]*N
dip_zi = [0.0]*N

ef_direct_s.update_dipoles_complex(dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi)
ef_ewald_s.update_dipoles_complex(dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi)

ef_direct_s.calculate()
ef_ewald_s.calculate()

E_direct_s = ef_direct_s.get_epoint_host()
E_ewald_s = ef_ewald_s.get_epoint_host()

print("E_direct_s (N,3):\n", E_direct_s)
print("E_ewald_s (N,3):\n", E_ewald_s)
print("Difference:\n", E_direct_s - E_ewald_s)
