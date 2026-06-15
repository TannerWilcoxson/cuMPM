import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma=681, omega_p=12313, eps_inf=4):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

def make_hexagonal_grid(Nx, Ny, spacing):
    """Generate 2D hexagonal lattice coordinates on a Nx x Ny grid."""
    positions = []
    for j in range(Ny):
        for i in range(Nx):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * (np.sqrt(3) / 2) * spacing
            positions.append([x, y, 0.0])
    return np.array(positions)

# Parameters
radius = 5.0
eps_m = 2.13
omega = np.linspace(1000, 7000, 80)
eps_p = drude_dielectric(omega)

Nx = 100
Ny = 100
N = Nx * Ny  # Exactly 10,000 particles
area_fractions = [0.1, 0.3, 0.5, 0.7]

mono_times = []
poly_times = []

print(f"Starting Field Implementation Benchmark for N = {N} particles (80 wavenumbers, BiCGSTAB)...")
print(f"{'Area Frac':<12}{'Monodisperse (s)':<18}{'Polydisperse (s)':<18}")
print("-" * 52)

# Warmup run to initialize CUDA context, CUFFT plans, etc.
# Use 1000 particles at 0.1 area fraction for quick warmup
warmup_spacing = radius * np.sqrt(2 * np.pi / (np.sqrt(3) * 0.1))
warmup_pos = make_hexagonal_grid(50, 20, warmup_spacing)
warmup_box = np.array([
    50 * warmup_spacing,
    20 * warmup_spacing * (np.sqrt(3) / 2),
    1000.0
])
# Warmup monodisperse field implementation
warmup_solver_mono = cuMPM.dipole_solver(
    box=warmup_box, eps_p=eps_p, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True,
    solver_type="bicgstab", field_type="monodisperse"
)
warmup_solver_mono.compute(warmup_pos)

# Warmup polydisperse field implementation
warmup_solver_poly = cuMPM.dipole_solver(
    box=warmup_box, eps_p=eps_p, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True,
    solver_type="bicgstab", field_type="polydisperse"
)
warmup_solver_poly.compute(warmup_pos)

# Run benchmark across area fractions
for phi in area_fractions:
    # Compute spacing for the target area fraction
    spacing = radius * np.sqrt(2 * np.pi / (np.sqrt(3) * phi))
    pos = make_hexagonal_grid(Nx, Ny, spacing)
    
    box = np.array([
        Nx * spacing,
        Ny * spacing * (np.sqrt(3) / 2),
        1000.0
    ])
    
    # --- Benchmark Monodisperse Field ---
    solver_mono = cuMPM.dipole_solver(
        box=box, eps_p=eps_p, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True,
        solver_type="bicgstab", field_type="monodisperse"
    )
    t0 = time.time()
    solver_mono.compute(pos)
    t_mono = time.time() - t0
    mono_times.append(t_mono)
    
    # --- Benchmark Polydisperse Field ---
    solver_poly = cuMPM.dipole_solver(
        box=box, eps_p=eps_p, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True,
        solver_type="bicgstab", field_type="polydisperse"
    )
    t0 = time.time()
    solver_poly.compute(pos)
    t_poly = time.time() - t0
    poly_times.append(t_poly)
    
    print(f"{phi:<12.1f}{t_mono:<18.4f}{t_poly:<18.4f}")

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(area_fractions, mono_times, 'o-', color='blue', linewidth=2, markersize=8, label="Monodisperse Field")
plt.plot(area_fractions, poly_times, 's--', color='red', linewidth=2, markersize=8, label="Polydisperse Field")

plt.title(f"cuMPM Field Implementation Runtime vs. Area Fraction (N = {N}, BiCGSTAB)", fontsize=12, fontweight='bold')
plt.xlabel("Area Fraction ($\phi$)", fontsize=11)
plt.ylabel("Execution Time (seconds)", fontsize=11)
plt.grid(True, which="both", linestyle=":", alpha=0.5)
plt.legend(frameon=True, shadow=True, fontsize=11)
plt.tight_layout()

# Save plot
os.makedirs("benchmarks/2d_lattice", exist_ok=True)
output_png = "benchmarks/2d_lattice/field_comparison.png"
plt.savefig(output_png, dpi=150)
print(f"Saved benchmark timing plot to {output_png}")
