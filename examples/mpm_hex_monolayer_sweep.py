import time
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

# 1. System and Material Parameters
d_opt = 10.0   # Optical diameter, nm
R = 0.5 * d_opt # Radius 'a'
eps_m = 1.0    # Surrounding medium permittivity (vacuum)

# Wavenumber range (120 points, cm^-1)
omega_orig = np.linspace(3000, 9000, 120)

# Convert wavenumber from cm^-1 to nm^-1 for the solver
omega_nm = omega_orig / 1e7

# Drude parameters for Tin-doped Indium Oxide (ITO)
omega_p = 11886
gamma = 845
eps_inf = 2.25
eps_p = drude_dielectric(omega_orig, gamma, omega_p, eps_inf)

# Calculate analytical single particle polarizability and extinction
# alpha_single = 4 * pi * eps_m * R^3 * (eps_p - eps_m) / (eps_p + 2 * eps_m)
alpha_single = 4.0 * np.pi * eps_m * (R**3) * (eps_p - eps_m) / (eps_p + 2.0 * eps_m)
k_vals = 2.0 * np.pi * omega_nm * np.sqrt(eps_m)
c_ext_single = 4.0 * np.pi * k_vals * np.imag(alpha_single)

# Incident plane wave parameters (Normal Incidence)
theta_deg = 0.0  # Angle of incidence in degrees
theta = np.radians(theta_deg)
k_dir = np.array([np.sin(theta), 0.0, np.cos(theta)]) # Propagation direction in xz-plane
pol_dir = np.array([0.0, 1.0, 0.0]) # Polarization direction (s-polarized, along y)

# 2. Sweep over L/a ratios
ratios = [2, 3, 4]
results = {}
peak_info = {}

# Set up matplotlib style for premium aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)

# Harmonious, modern color palette for the curves
colors = {
    2: '#2a9d8f',  # Teal
    3: '#e76f51',  # Terracotta / Warm Orange
    4: '#264653'   # Dark Slate
}

for ratio in ratios:
    # Lattice constant L = ratio * R
    d = ratio * R
    
    # 2. Periodic Hexagonal Lattice Generation
    # We choose Nx and Ny such that 2 * Nx * Ny = 4000
    Nx = 50
    Ny = 40
    num_nc = 2 * Nx * Ny  # Exactly 4000
    
    pos = []
    for i in range(Nx):
        for j in range(Ny):
            # Sublattice A
            pos.append([i * d, j * np.sqrt(3.0) * d, 0.0])
            # Sublattice B
            pos.append([(i + 0.5) * d, (j + 0.5) * np.sqrt(3.0) * d, 0.0])
    pos = np.array(pos)
    
    # Define periodic 2D box dimensions (z-dimension = 50R)
    Lx = Nx * d
    Ly = Ny * np.sqrt(3.0) * d
    Lz = 50.0 * R
    box = np.array([Lx, Ly, Lz])
    
    print(f"\n==================================================")
    print(f"Running simulation for L/a = {ratio} (Plane Wave, Ewald)")
    print(f"Lattice Constant Lc (d): {d:.2f} nm, Radius a (R): {R:.2f} nm")
    print(f"Number of NCs in supercell: {num_nc}")
    print(f"Box dimensions: {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} nm")
    print(f"==================================================")
    
    t0 = time.time()
    c_ext = np.zeros(len(omega_orig))
    
    # Loop over wavelengths (Python loop is highly optimized as it avoids quadratic solver calls)
    for iw, w_nm in enumerate(omega_nm):
        eps_w = eps_p[iw]
        k_val = k_vals[iw]
        
        # Calculate incident plane wave field at particle positions
        k_vec = k_dir * k_val
        phases = np.dot(pos, k_vec)
        E_inc_w = pol_dir[None, :] * np.exp(1j * phases)[:, None]
        E0_w = E_inc_w.ravel()
        
        # Solve for dipoles at this wavelength
        solver = cuMPM.dipole_solver(
            box=box,
            eps_p=eps_w,
            radius=R,
            eps_m=eps_m,
            xi=0.5,
            tol=1e-3,
            quiet=True,
            solver_type="bicgstab",
            field_type="monodisperse",
            E0=E0_w
        )
        solver.compute(pos)
        dips_w = solver.get_dipoles(physical=True) # shape: (num_nc, 3)
        
        # Calculate Extinction Cross-Section (C_ext)
        proj = np.sum(np.conj(E_inc_w) * dips_w, axis=1)
        c_ext[iw] = 4.0 * np.pi * k_val * np.sum(np.imag(proj))
        
        if (iw + 1) % 20 == 0 or iw == len(omega_nm) - 1:
            print(f"  Processed wavelength {iw+1}/{len(omega_nm)}...")
            
    t_run = time.time() - t0
    print(f"Completed ratio sweep in {t_run:.2f} seconds.")
    
    results[ratio] = c_ext
    omega_ev = omega_orig / 8065.54
    
    # Save raw data for this ratio
    np.savez(f'extinction_data_ratio_{ratio}.npz', 
             wavenumber_cm=omega_orig, 
             energy_ev=omega_ev, 
             extinction_cs=c_ext,
             extinction_cs_single=c_ext_single)
    
    # Analyze peak
    peak_idx = np.argmax(c_ext)
    peak_wavenumber = omega_orig[peak_idx]
    peak_energy = omega_ev[peak_idx]
    peak_val = c_ext[peak_idx]
    peak_info[ratio] = (peak_wavenumber, peak_val)
    print(f"L/a = {ratio} | LSPR Extinction Peak at: {peak_wavenumber:.0f} cm^-1 ({peak_energy:.3f} eV) with value {peak_val:.2f} nm^2")

    # Plot this curve in wavenumber
    ax.plot(omega_orig, c_ext, color=colors[ratio], linewidth=2.5, 
            label=f"$L/a = {ratio}$ (Peak: {peak_wavenumber:.0f} cm$^{{-1}}$)")
    
    # Add a subtle vertical line at the peak
    ax.axvline(x=peak_wavenumber, color=colors[ratio], linestyle=':', alpha=0.6, linewidth=1.2)

# Plot single particle extinction (scaled by N to represent non-interacting limit)
peak_single_idx = np.argmax(c_ext_single)
peak_single_wavenumber = omega_orig[peak_single_idx]

ax.plot(omega_orig, num_nc * c_ext_single, color='#8d99ae', linestyle='--', linewidth=2, 
        label=f"Non-interacting NCs ($N \\times C_{{\\text{{ext}}}}^{{\\text{{single}}}}$, Peak: {peak_single_wavenumber:.0f} cm$^{{-1}}$)")

# Save combined results
np.savez('extinction_sweep_data.npz', 
         ratios=ratios, 
         wavenumber_cm=omega_orig, 
         extinction_cs_single=c_ext_single,
         **{f'extinction_cs_{r}': results[r] for r in ratios})

# Formatting the plot
ax.set_title("Extinction Cross Section vs. Lattice-to-Radius Ratio ($L/a$)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
ax.set_ylabel("Extinction Cross Section ($C_{\\text{ext}}$, nm$^2$)", fontsize=12)
ax.grid(True, linestyle=':', alpha=0.5)

# High energy (9000 cm^-1) on the left, low energy (3000 cm^-1) on the right
ax.set_xlim(np.max(omega_orig), np.min(omega_orig))

ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)

# Annotate peaks
for ratio, (p_wavenumber, p_val) in peak_info.items():
    ax.annotate(f"{p_wavenumber:.0f} cm$^{{-1}}$",
                xy=(p_wavenumber, p_val),
                xytext=(p_wavenumber - 600, p_val * 0.95),
                arrowprops=dict(arrowstyle="->", color=colors[ratio], alpha=0.7),
                fontsize=9, fontweight='bold', color=colors[ratio])

plt.tight_layout()
plt.savefig("extinction_hex_monolayer_sweep.png", dpi=200)
print("\nSaved combined extinction cross section plot to extinction_hex_monolayer_sweep.png")
