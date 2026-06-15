import time
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

# Parameters
d = 12.0  # Spacing between nearest neighbor Na+ and Cl- particles in nm (lattice parameter a = 24.0 nm)
eps_m = 2.13  # Surrounding medium dielectric constant

# Wavenumber range (80 points)
omega = np.linspace(1000, 7000, 80)

# Setup the two different Drude dielectrics (matching mixed_eps_mpm.py)
# Type 1 (Na+ sites): Original Drude parameters
omega_p1 = 12313
gamma1 = 681
eps_inf1 = 4
eps_p1 = drude_dielectric(omega, gamma1, omega_p1, eps_inf1)

# Type 2 (Cl- sites): Alternative Drude parameters
omega_p2 = 8000
gamma2 = 400
eps_inf2 = 2
eps_p2 = drude_dielectric(omega, gamma2, omega_p2, eps_inf2)

# Create a 6x6x6 simple cubic grid (216 particles)
N = 6
A = np.arange(0, N * d, d)
z, y, x = np.meshgrid(A, A, A, indexing='ij')

# Reshape into list of coordinates
pos = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
num_particles = N**3
box = np.array([N * d, N * d, N * d])

# Initialize arrays for polydisperse radii and mixed permittivities
radii = np.zeros(num_particles)
eps_p_mixed = np.zeros((len(omega), num_particles), dtype=complex)

# Setup NaCl checkerboard structure based on grid index sum (ix + iy + iz)
# ix, iy, iz can be recovered from the grid spacing
for idx in range(num_particles):
    px, py, pz = pos[idx]
    ix = int(round(px / d))
    iy = int(round(py / d))
    iz = int(round(pz / d))
    
    # Check parity of coordinate sum
    if (ix + iy + iz) % 2 == 0:
        # Na+ site
        radii[idx] = 5.0  # radius = 5.0 nm
        eps_p_mixed[:, idx] = eps_p1
    else:
        # Cl- site
        radii[idx] = 3.0  # radius = 3.0 nm
        eps_p_mixed[:, idx] = eps_p2

print(f"Running 3D Polydisperse NaCl Crystal Solver for {num_particles} particles...")

# Run cuMPM (GPU-resident complex solver)
t0 = time.time()
mpm_cu = cuMPM.dipole_solver(box=box, eps_p=eps_p_mixed, radius=radii, eps_m=eps_m, tol=1e-3, solver_type="gmres")
mpm_cu.compute(pos)
t_cu = time.time() - t0
alpha_cu = mpm_cu.get_eff_polarizability()

print(f"cuMPM ran successfully in: {t_cu:.2f}s")

# Setup plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Imaginary Part
axes[0].plot(omega, alpha_cu.imag[:, 0, 0], 'b-', linewidth=2, label="xx (in-plane)")
axes[0].plot(omega, alpha_cu.imag[:, 1, 1], 'r--', linewidth=2, label="yy (in-plane)")
axes[0].plot(omega, alpha_cu.imag[:, 2, 2], 'g-.', linewidth=2, label="zz (out-of-plane)")
axes[0].set_title("Imaginary Polarizability", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Wavenumber $\\omega$ (cm$^{-1}$)")
axes[0].set_ylabel("Im($\\alpha_{\\text{eff}}$)")
axes[0].set_xlim(np.flip(axes[0].get_xlim()))  # Flip x-axis
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(frameon=True, shadow=True)

# Plot Real Part
axes[1].plot(omega, alpha_cu.real[:, 0, 0], 'b-', linewidth=2, label="xx (in-plane)")
axes[1].plot(omega, alpha_cu.real[:, 1, 1], 'r--', linewidth=2, label="yy (in-plane)")
axes[1].plot(omega, alpha_cu.real[:, 2, 2], 'g-.', linewidth=2, label="zz (out-of-plane)")
axes[1].set_title("Real Polarizability", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Wavenumber $\\omega$ (cm$^{-1}$)")
axes[1].set_ylabel("Re($\\alpha_{\\text{eff}}$)")
axes[1].set_xlim(np.flip(axes[1].get_xlim()))  # Flip x-axis
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(frameon=True, shadow=True)

plt.suptitle(f"3D NaCl Crystal Lattice Polarizability ({num_particles} Particles)", fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
# output_png = "polydisperse_3d_polarizability.png"
# plt.savefig(output_png, dpi=150)
