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
    split_dist=split_dist, N_split=800, field_type="auto", quiet=False
)

# 3. Single Impact Parameter calculation at x = 4.67 nm
epos_sweep = np.array([[4.67, 0.0]])

print(f"Running EELS for Unsplit at x = 4.67 nm ({len(omega)} wavelengths)...")
t0 = time.time()
solver_unsplit.compute(epos_sweep, pos)
eels_unsplit = solver_unsplit.get_eels() # Shape: (120,)
print(f"Unsplit finished in {time.time() - t0:.2f}s")

print(f"Running EELS for Split (split_dist = 1.2R = {split_dist} nm, N_split = 100)...")
t0 = time.time()
solver_split.compute(epos_sweep, pos)
eels_split = solver_split.get_eels() # Shape: (120,)
print(f"Split finished in {time.time() - t0:.2f}s")

# 4. Plotting Results
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(omega, eels_unsplit, 'k--', label='Unsplit', linewidth=2)
ax1.plot(omega, eels_split, 'r-', label='Split (split_dist=1.2R)', linewidth=2)
ax1.axvline(omega_p, color='blue', linestyle=':', label='Bulk LSPR Peak')
ax1.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
ax1.set_ylabel('EELS Probability', fontsize=11)
ax1.set_title('EELS Spectrum at Impact Parameter x = 4.67 nm', fontsize=12, fontweight='bold')
ax1.legend(frameon=True, facecolor='white', edgecolor='lightgray')

plt.tight_layout()
plt.savefig('eels_single_ito_split.png', dpi=300)
print("Saved plot to eels_single_ito_split.png")
