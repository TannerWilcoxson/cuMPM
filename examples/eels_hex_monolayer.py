import time
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

# 1. System and Material Parameters
d = 12.0       # Center-to-center particle spacing, nm
d_opt = 10.0   # Optical diameter, nm
R = 0.5 * d_opt
eps_m = 1.0    # Surrounding medium permittivity (vacuum)
v = 0.5        # Electron velocity as a fraction of c (80 keV)

# Wavenumber range (100 points, cm^-1)
omega = np.linspace(3000, 9000, 120)

# Drude parameters for Tin-doped Indium Oxide (ITO)
omega_p = 11886
gamma = 845
eps_inf = 2.25
eps_p = drude_dielectric(omega, gamma, omega_p, eps_inf)

# Convert wavenumber from cm^-1 to nm^-1 for the solver
omega_nm = omega / 1e7

# 2. Hexagonal Lattice Generation
# Define hexagonal lattice basis vectors
a1 = np.array([d, 0.0, 0.0])
a2 = np.array([0.5 * d, 0.5 * np.sqrt(3) * d, 0.0])

# Generate a grid of points
grid_range = np.arange(-130, 131)
all_points = []
for i in grid_range:
    for j in grid_range:
        pt = i * a1 + j * a2
        all_points.append(pt)
all_points = np.array(all_points)

# Define the circumcenter of the triangle formed by three nearest neighbors:
# (0, 0, 0), (d, 0, 0), and (0.5 * d, 0.5 * np.sqrt(3) * d, 0)
centroid = np.array([0.5 * d, (np.sqrt(3) / 6.0) * d, 0.0])

# Calculate distance of each point to the centroid
dists = np.linalg.norm(all_points - centroid, axis=1)

# Keep exactly the 40000 closest particles to form a symmetric patch
num_nc = 4000
closest_indices = np.argsort(dists)[:num_nc]
pos = all_points[closest_indices]

# Shift the system so that the ebeam path (centroid) is exactly at (0, 0, 0)
pos = pos - centroid
e_pos = np.array([0.0, 0.0])

# Define the bounding box size (large enough to envelope the entire cluster)
max_coord = np.max(np.abs(pos))
box_dim = 2.0 * max_coord + 5.0 * d
box = np.array([box_dim, box_dim, box_dim])

print(f"Generated hexagonal monolayer patch of {num_nc} ITO nanocrystals.")
print(f"E-beam position is centered at (0, 0) equidistant from the nearest three NCs.")
print(f"Box dimensions: {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} nm")

# 3. Run C++ cuMPM EELS Solver
print("Running cuMPM GPU EELS calculation...")
t0 = time.time()
eels_solver = cuMPM.EELS(
    box=box,
    eps_p=eps_p,
    omega=omega_nm,
    v=v,
    eps_m=eps_m,
    radius=R,
    xi=0.5,
    tol=1e-3,
    field_type="direct",
    solver_type="bicgstab"
)
eels_solver.compute(epos=e_pos, positions=pos)
t_run = time.time() - t0
print(f"cuMPM GPU solver completed in {t_run:.2f} seconds.")

# Retrieve EELS spectrum and dipoles
eels_spectrum = eels_solver.get_eels()

# 4. Premium Aesthetic Plotting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Custom color palette and font settings
primary_color = '#005f73' # Sleek deep teal
accent_color = '#ae2012'  # Muted warm red for peaks

# Plot EELS spectrum
omega = omega/8065.54

ax.plot(omega, eels_spectrum, color=primary_color, linewidth=2.5, label="EELS Probability")

# Peak finding for annotation
peak_idx = np.argmax(eels_spectrum)
peak_omega = omega[peak_idx]
peak_val = eels_spectrum[peak_idx]

ax.axvline(x=peak_omega, color=accent_color, linestyle='--', alpha=0.7, linewidth=1.5)
ax.annotate(f"LSPR Peak: {peak_omega:.0f} cm$^{{-1}}$",
            xy=(peak_omega, peak_val),
            xytext=(peak_omega + 400, peak_val * 0.9),
            arrowprops=dict(facecolor=accent_color, arrowstyle="->", connectionstyle="arc3,rad=.2"),
            fontsize=10, fontweight='bold', color=accent_color)

print(f"MAX EELS VALUE DIRECT: {peak_val}")
np.savez('direct_eels_data.npz', wls=omega, loss_eels=eels_spectrum)

# Formatting
ax.set_title(f"EELS Probability Spectrum for 2D ITO Hexagonal Monolayer ({num_nc} NCs)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Wavenumber $\\omega$ (cm$^{-1}$)", fontsize=11)
ax.set_ylabel("EELS Probability (arb. units)", fontsize=11)
#ax.set_xlim(np.max(omega), np.min(omega)) # Spectral display: high to low wavenumber
ax.grid(True, linestyle=':', alpha=0.5)

# Inset plot: NC patch configuration
ax_inset = fig.add_axes([0.15, 0.15, 0.25, 0.25])
ax_inset.scatter(pos[:, 0], pos[:, 1], s=0.8, color='#94d2bd', alpha=0.8, edgecolors='none')
ax_inset.scatter(0, 0, s=25, color=accent_color, marker='x', label='E-beam')
ax_inset.set_aspect('equal')
ax_inset.set_title("NC Patch & E-beam", fontsize=8, fontweight='bold')
ax_inset.axis('off')
ax_inset.legend(loc='upper right', fontsize=6, frameon=True)

plt.savefig(f"eels_hex_monolayer_{num_nc}.png")
print(f"Saved EELS probability plot to eels_hex_monolayer_{num_nc}.png")
