import time
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

# 1. System and Material Parameters
d = 27.6       # Center-to-center particle spacing, nm (lattice constant)
d_opt = 25   # Optical diameter, nm
R = 0.5 * d_opt
eps_m = 1.0    # Surrounding medium permittivity (vacuum)
v = 0.45        # Electron velocity as a fraction of c (80 keV)

# Wavenumber range (cm^-1)
omega = np.linspace(3000, 9000, 100)
omega_nm = omega / 1e7

# Drude parameters for Tin-doped Indium Oxide (ITO)
omega_p = 11895
gamma = 863
eps_inf = 2.25
eps_p = drude_dielectric(omega, gamma, omega_p, eps_inf)

# 2. Hexagonal Supercell Generation
# To satisfy the Ewald solver's box size constraint (L > 2 * r_cut), we create a larger 
# periodic rectangular supercell containing a hexagonal lattice of nanocrystals.
# A primitive rectangular cell of a hex lattice has dimensions d by sqrt(3)*d and contains 2 particles.
Nx_super = 6
Ny_super = 6
box_x = Nx_super * d
box_y = Ny_super * np.sqrt(3) * d
box_z = d * 20.0
box = np.array([box_x, box_y, box_z])

pos = []
for i in range(Nx_super):
    for j in range(Ny_super):
        # Particle 1 in primitive rectangular cell
        pos.append([i * d, j * np.sqrt(3) * d, 0.0])
        # Particle 2 in primitive rectangular cell
        pos.append([(i + 0.5) * d, (j + 0.5) * np.sqrt(3) * d, 0.0])
pos = np.array(pos)

# Center the supercell at the origin
pos[:, 0] -= box_x / 2.0
pos[:, 1] -= box_y / 2.0

e_pos = np.array([0.5 * d, (np.sqrt(3) / 6.0) * d])


print(f"Generated a {Nx_super}x{Ny_super} rectangular supercell of a hexagonal lattice ({len(pos)} ITO nanocrystals).")
print(f"Lattice constant: {d} nm. E-beam position is at ({e_pos[0]}, {e_pos[1]}).")
print(f"Supercell Box dimensions: {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} nm")

# 3. Run C++ cuMPM EELS Solver with ASM enabled
# Because our supercell is large, the folded Brillouin zone is small. We need fewer k-points!
Nx = 20
Ny = 20
print(f"Running cuMPM GPU EELS calculation with ASM (Nx={Nx}, Ny={Ny})...")
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
    field_type="monodisperse",  # Must be an Ewald field for periodic bounds
    solver_type="gmres",
    asm_flag=True,              # Enable Array Scanning Method!
    asm_Nx=Nx,                  # Number of k-points in x-direction
    asm_Ny=Ny                   # Number of k-points in y-direction
)
eels_solver.compute(epos=e_pos, positions=pos)
t_run = time.time() - t0
print(f"cuMPM GPU solver completed in {t_run:.2f} seconds.")

# Retrieve EELS spectrum
eels_spectrum = eels_solver.get_eels()

# 4. Plotting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

# Convert x-axis to eV for standard EELS units (1 eV ~ 8065.5 cm^-1)
omega_ev = omega / 8065.54

ax.plot(omega_ev, eels_spectrum, lw=2.5, color='#005f73', label=f'ASM EELS (Nx={Nx}, Ny={Ny})')

ax.set_xlabel('Energy Loss (eV)', fontsize=12, fontweight='bold', color='#333333')
ax.set_ylabel('EELS Probability ($\\Gamma_{EELS}$)', fontsize=12, fontweight='bold', color='#333333')
ax.set_title('Infinite 2D Square Array EELS Spectra via Array Scanning Method', fontsize=14, fontweight='bold', color='#111111', pad=15)

ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(frameon=True, fancybox=True, shadow=True, borderpad=1, fontsize=11)

plt.tight_layout()
plt.savefig(f'eels_asm_spectra_{len(pos)}.png', dpi=300, bbox_inches='tight')
print(f"Saved plot to eels_asm_spectra_{len(pos)}.png")
print(f"MAX EELS VALUE ASM: {np.max(eels_spectrum)}")
np.savez('asm_eels_data.npz', wls=omega_ev, loss_eels=eels_spectrum)
