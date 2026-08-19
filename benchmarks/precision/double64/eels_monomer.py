import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cuMPM


def drude_dielectric(omega, gamma, omega_p, eps_inf):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)


def main():
    # -------------------------------------------------------------------------
    # 1. System and Material Parameters
    # -------------------------------------------------------------------------
    box = [100.0, 100.0, 100.0]
    radius = 12.5  # 25 nm diameter NC (radius = 12.5 nm)
    eps_m = 1.0    # Vacuum
    v = 0.5        # 80 keV electron beam velocity (fraction of c)

    # Single NC at origin
    pos = np.array([[0.0, 0.0, 0.0]])

    # 80 energy points between 0.3 and 1.1 eV
    num_energy_pts = 80
    omega_ev = np.linspace(0.3, 1.1, num_energy_pts)
    omega_cm = omega_ev * 8065.544
    omega_nm = omega_cm / 1e7

    # Drude parameters for Tin-doped Indium Oxide (ITO)
    omega_p = 11886.0  # cm^-1 (~1.474 eV)
    gamma = 845.0      # cm^-1
    eps_inf = 2.25
    eps_p = drude_dielectric(omega_cm, gamma, omega_p, eps_inf)

    # -------------------------------------------------------------------------
    # 2. Impact Parameters (in nm)
    #    - Center: (0.001, 0.0) nm
    #    - Inner edge: (11.0, 0.0) nm (1.5 nm inside boundary)
    #    - Outer edge: (13.5, 0.0) nm (1.0 nm outside boundary)
    # -------------------------------------------------------------------------
    epos_sweep = np.array([
        [0.001, 0.0],  # Center
        [11.0, 0.0],   # Inner Edge
        [13.5, 0.0]    # Outer Edge
    ])
    epos_labels = [
        "Center (x = 0.001 nm)",
        "Inner Edge (x = 11.0 nm)",
        "Outer Edge (x = 13.5 nm)"
    ]
    colors = ["#005f73", "#ca6702", "#ae2012"]
    linestyles = ["-", "--", "-."]

    # -------------------------------------------------------------------------
    # 3. Solver Setup: FCC Lattice Dipole Solver (N_split = 320000)
    # -------------------------------------------------------------------------
    N_split = 320000
    split_dist = 2.0 * radius  # Active for all impact parameters within 2R
    split_method = "fcc"

    print(f"Setting up cuMPM EELS solver (Radius = {radius} nm, FCC Lattice, Dipole Only, N_split = {N_split})...")

    # Split Dipole (FCC, N_split = 320000)
    solver_split_dipole = cuMPM.EELS(
        box=box,
        eps_p=eps_p,
        omega=omega_nm,
        v=v,
        eps_m=eps_m,
        radius=radius,
        solve_quadrupoles=False,
        split_dist=split_dist,
        N_split=N_split,
        split_method=split_method,
        field_type="direct",
        precision="fp64",
        quiet=False
    )

    # -------------------------------------------------------------------------
    # 4. Compute EELS Spectra
    # -------------------------------------------------------------------------
    print(f"\nComputing FCC Split Dipole EELS (N_split={N_split}, {num_energy_pts} energy points)...")
    t0 = time.time()
    solver_split_dipole.compute(epos_sweep, pos)
    eels_split_dipole_raw = solver_split_dipole.get_eels()
    print(f"FCC Split Dipole finished in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # Redimensionalization formula (accounting for E-beam prefactor without 4pi):
    # F_redim = 8 * pi^2 * a * e^2 / (hbar^2 * c^2 * eps_m * eps_0)
    # Note: a is the sub-particle radius R_sub = R / N_split^(1/3) used by cuMPM
    # -------------------------------------------------------------------------
    R_sub_nm = radius / (N_split ** (1.0 / 3.0)) if (N_split > 1) else radius
    a_m = R_sub_nm * 1e-9                # Length scale a in meters

    e_C = 1.602176634e-19                # Coulomb
    hbar_J_s = 1.054571817e-34           # J * s
    hbar_eV_s = 6.582119569e-16          # eV * s
    c_m_s = 2.99792458e8                 # m/s
    eps_0 = 8.8541878128e-12             # F/m

    # F_redim in eV^-1 per code unit (~ 0.024704 eV^-1)
    F_redim = (8.0 * (np.pi**2) * a_m * (e_C**2)) / (hbar_J_s * hbar_eV_s * (c_m_s**2) * eps_m * eps_0)
    eels_split_dipole = eels_split_dipole_raw * F_redim

    # -------------------------------------------------------------------------
    # 5. Theoretical Reference Energies
    # -------------------------------------------------------------------------
    ev_bulk = (omega_p / np.sqrt(eps_inf)) / 8065.544
    ev_lspr = (omega_p / np.sqrt(eps_inf + 2.0 * eps_m)) / 8065.544

    # -------------------------------------------------------------------------
    # 6. Save Numerical Data
    # -------------------------------------------------------------------------
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, "eels_monomer_data.npz")
    np.savez(
        data_path,
        omega_ev=omega_ev,
        epos_sweep=epos_sweep,
        eels_split_dipole=eels_split_dipole,
        N_split=N_split,
        split_method=split_method,
        radius=radius,
        ev_bulk=ev_bulk,
        ev_lspr=ev_lspr
    )
    print(f"\nSaved calculation data to {data_path}")

    # -------------------------------------------------------------------------
    # 7. Visualization / Plotting (All on single axes)
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    for idx, label in enumerate(epos_labels):
        ax.plot(
            omega_ev,
            eels_split_dipole[idx],
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=2.2,
            label=label
        )

    # Reference plasmon lines
    ax.axvline(ev_lspr, color="#7209b7", linestyle=":", linewidth=1.4, label=f"Sphere LSPR ({ev_lspr:.3f} eV)")
    ax.axvline(ev_bulk, color="#3a86ff", linestyle="-.", linewidth=1.4, label=f"Bulk Plasmon ({ev_bulk:.3f} eV)")

    ax.set_xlabel("Energy Loss (eV)", fontsize=12, fontweight="bold")
    ax.set_ylabel("EELS Probability $\\Gamma(E)$ (eV$^{-1}$)", fontsize=12, fontweight="bold")
    ax.set_title(f"Single ITO Nanocrystal (25 nm) EELS Spectra ($N_{{split}} = {N_split}$, FCC Placement)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=10, loc="upper right")

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "eels_monomer.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved EELS plot to {plot_path}")


if __name__ == "__main__":
    main()
