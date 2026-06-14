import time
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

# Parameters
d = 11.7  # Particle diameter, nm
d_opt = 10.0  # Particle optical diameter, nm
eps_m = 2.13  # Medium dielectric constant

# Wavenumber range (80 points)
omega = np.linspace(1000, 7000, 80)

# Setup checkerboard pattern of two different Drude dielectrics
# Type 1: Original Drude parameters
omega_p1 = 12313
gamma1 = 681
eps_inf1 = 4
eps_p1 = drude_dielectric(omega, gamma1, omega_p1, eps_inf1)

# Type 2: Alternative Drude parameters
omega_p2 = 8000
gamma2 = 400
eps_inf2 = 2
eps_p2 = drude_dielectric(omega, gamma2, omega_p2, eps_inf2)

# Create 12x12 square lattice (144 particles)
N = 20
L = d * N
A = np.arange(0, L, d)
y, x = np.meshgrid(A, A)
pos = np.array([x, y, np.zeros_like(x)]).T
pos = pos.reshape(N**2, 3)
box = np.array([L, L, 50 * d])

num_particles = N**2
eps_p_mixed = np.zeros((len(omega), num_particles), dtype=complex)

# Assign dielectrics in checkerboard pattern
for iy in range(N):
    for ix in range(N):
        idx = iy * N + ix
        if (ix + iy) % 2 == 0:
            eps_p_mixed[:, idx] = eps_p1
        else:
            eps_p_mixed[:, idx] = eps_p2

print(f"Running mixed-dielectric solver for {num_particles} particles...")

# Run cuMPM (GPU)
t0 = time.time()
mpm_cu = cuMPM.dipole_solver(box, eps_p_mixed, radius=d_opt/2, eps_m=eps_m, tol=1e-3)
mpm_cu.compute(pos)
t_cu = time.time() - t0
alpha_cu = mpm_cu.get_eff_polarizability()

print(f"cuMPM ran in: {t_cu:.2f}s")

# Setup plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Imaginary Part
axes[0].plot(omega, alpha_cu.imag[:, 0, 0], 'b-', linewidth=2, label="In-plane (xx)")
axes[0].plot(omega, alpha_cu.imag[:, 2, 2], 'g-', linewidth=2, label="Out-of-plane (zz)")
axes[0].set_title("Imaginary Polarizability", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Wavenumber $\\omega$ (cm$^{-1}$)")
axes[0].set_ylabel("Im($\\alpha_{\\text{eff}}$)")
axes[0].set_xlim(np.flip(axes[0].get_xlim()))  # Flip x-axis
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(frameon=True, shadow=True)

# Plot Real Part
axes[1].plot(omega, alpha_cu.real[:, 0, 0], 'b-', linewidth=2, label="In-plane (xx)")
axes[1].plot(omega, alpha_cu.real[:, 2, 2], 'g-', linewidth=2, label="Out-of-plane (zz)")
axes[1].set_title("Real Polarizability", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Wavenumber $\\omega$ (cm$^{-1}$)")
axes[1].set_ylabel("Re($\\alpha_{\\text{eff}}$)")
axes[1].set_xlim(np.flip(axes[1].get_xlim()))  # Flip x-axis
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(frameon=True, shadow=True)

plt.suptitle(f"Mixed-Dielectric Square Lattice Polarizability ({num_particles} Particles)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("mixed_eps_polarizability.png", dpi=300)
print("Saved polarizability plot to mixed_eps_polarizability.png")
