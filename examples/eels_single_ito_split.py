import time
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

# 1. System and Material Parameters
box = [100.0, 100.0, 100.0]
radius = 5.0   # 10 nm diameter
eps_m = 1.0    # vacuum
v = 0.5        # 80 keV

# Wavenumber range (cm^-1)
omega = np.linspace(3000, 9000, 120)
omega_nm = omega / 1e7

# Drude parameters for Tin-doped Indium Oxide (ITO)
omega_p = 11886
gamma = 845
eps_inf = 2.25
eps_p = drude_dielectric(omega, gamma, omega_p, eps_inf)

# Single NC at origin
pos = np.array([[0.0, 0.0, 0.0]])

# 2. Set up EELS Solvers
# Case A: No splitting
solver_unsplit = cuMPM.EELS(
    box=box, eps_p=eps_p, omega=omega_nm, v=v, eps_m=eps_m, radius=radius,
    field_type="auto", quiet=False
)

# Case B: Splitting active near the edge/corner (split_dist = 1.2 * radius = 6.0 nm)
split_dist = 1.2 * radius
solver_split = cuMPM.EELS(
    box=box, eps_p=eps_p, omega=omega_nm, v=v, eps_m=eps_m, radius=radius,
    split_dist=split_dist, N_split=100, field_type="auto", quiet=False
)

# 3. Impact Parameter (epos) Sweep
# Sweep x from 0.0 nm (center) to 7.0 nm (past the 5.0 nm surface and 6.0 nm split limit)
sweep_x = np.linspace(0.0, 7.0, 25)
epos_sweep = np.column_stack([sweep_x, np.zeros_like(sweep_x)])

print(f"Running EELS sweep for Unsplit ({len(sweep_x)} epos points, {len(omega)} wavelengths)...")
t0 = time.time()
solver_unsplit.compute(epos_sweep, pos)
eels_unsplit = solver_unsplit.get_eels() # Shape: (25, 30)
print(f"Unsplit sweep finished in {time.time() - t0:.2f}s")

print(f"Running EELS sweep for Split (split_dist = 1.2R = {split_dist} nm, N_split = 100)...")
t0 = time.time()
solver_split.compute(epos_sweep, pos)
eels_split = solver_split.get_eels() # Shape: (25, 30)
print(f"Split sweep finished in {time.time() - t0:.2f}s")

# 4. Plotting Results
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Compare EELS spectra at an impact parameter near the edge (x = 4.8 nm)
idx_edge = np.argmin(np.abs(sweep_x - 4.8))
ax1.plot(omega, eels_unsplit[idx_edge], 'k--', label='Unsplit', linewidth=2)
ax1.plot(omega, eels_split[idx_edge], 'r-', label='Split (split_dist=1.2R)', linewidth=2)
ax1.axvline(omega_p, color='blue', linestyle=':', label='Bulk LSPR Peak')
ax1.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
ax1.set_ylabel('EELS Probability', fontsize=11)
ax1.set_title(f'EELS Spectrum at Impact Parameter x = {sweep_x[idx_edge]:.2f} nm', fontsize=12, fontweight='bold')
ax1.legend(frameon=True, facecolor='white', edgecolor='lightgray')

# Plot 2: Compare EELS probability vs impact parameter (x) at resonance peak (approx 6500 cm^-1)
peak_idx = np.argmax(eels_unsplit[idx_edge])
peak_omega = omega[peak_idx]

ax2.plot(sweep_x, eels_unsplit[:, peak_idx], 'k--', label='Unsplit', linewidth=2)
ax2.plot(sweep_x, eels_split[:, peak_idx], 'r-', label='Split (split_dist=1.2R)', linewidth=2)

# Visual indicators for NC radius and split regions
ax2.axvline(radius, color='gray', linestyle='-.', label='NC Surface (R = 5.0 nm)')
ax2.axvline(split_dist, color='red', linestyle=':', label=f'Split Cutoff (1.2R = {split_dist} nm)')
ax2.axvspan(0.0, split_dist, color='red', alpha=0.08, label='Splitting Active Region')

ax2.set_xlabel('Impact Parameter x (nm)', fontsize=11)
ax2.set_ylabel('EELS Probability', fontsize=11)
ax2.set_title(f'EELS Intensity vs Impact Parameter at {peak_omega:.0f} cm$^{{-1}}$', fontsize=12, fontweight='bold')
ax2.legend(frameon=True, facecolor='white', edgecolor='lightgray')

plt.tight_layout()
plt.savefig('eels_single_ito_split.png', dpi=300)
print("Saved plot to eels_single_ito_split.png")
