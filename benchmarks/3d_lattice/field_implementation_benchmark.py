import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma=681, omega_p=12313, eps_inf=4):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

def make_simple_cubic_grid_3d(Nx, Ny, Nz, spacing):
    """Generate 3D simple cubic grid coordinates."""
    positions = []
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                positions.append([i * spacing, j * spacing, k * spacing])
    return np.array(positions)

# Parameters
radius = 5.0
eps_m = 2.13
omega = np.linspace(1000, 7000, 80) # 80 wavenumbers as requested
eps_p = drude_dielectric(omega)

Nx = 50
Ny = 50
Nz = 40
N = Nx * Ny * Nz  # Exactly 100,000 particles
volume_fractions = [0.1, 0.3, 0.5, 0.7]

mono_times = []
poly_times = []

print(f"Starting 3D Field Implementation Benchmark for N = {N} particles (80 wavenumbers, BiCGSTAB)...")
print(f"{'Volume Frac':<15}{'Monodisperse (s)':<18}{'Polydisperse (s)':<18}")
print("-" * 52)

# Warmup run to initialize CUDA context, CUFFT plans, etc.
# Use Nx=10, Ny=10, Nz=10 -> 1000 particles for quick warmup at phi = 0.1
warmup_spacing = radius * (4 * np.pi / (3 * 0.1))**(1/3)
warmup_pos = make_simple_cubic_grid_3d(10, 10, 10, warmup_spacing)
warmup_box = np.array([
    10 * warmup_spacing,
    10 * warmup_spacing,
    10 * warmup_spacing
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

# Run benchmark across volume fractions
for phi in volume_fractions:
    # Compute spacing for the target 3D volume fraction
    spacing = radius * (4 * np.pi / (3 * phi))**(1/3)
    pos = make_simple_cubic_grid_3d(Nx, Ny, Nz, spacing)
    
    box = np.array([
        Nx * spacing,
        Ny * spacing,
        Nz * spacing
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
    
    print(f"{phi:<15.1f}{t_mono:<18.4f}{t_poly:<18.4f}")

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(volume_fractions, mono_times, 'o-', color='blue', linewidth=2, markersize=8, label="Monodisperse Field")
plt.plot(volume_fractions, poly_times, 's--', color='red', linewidth=2, markersize=8, label="Polydisperse Field")

plt.title(f"cuMPM 3D Field Runtime vs. Volume Fraction (N = {N}, 80 Wavenumbers)", fontsize=12, fontweight='bold')
plt.xlabel("Volume Fraction ($\phi$)", fontsize=11)
plt.ylabel("Execution Time (seconds)", fontsize=11)
plt.grid(True, which="both", linestyle=":", alpha=0.5)
plt.legend(frameon=True, shadow=True, fontsize=11)
plt.tight_layout()

# Save plot
os.makedirs("benchmarks/3d_lattice", exist_ok=True)
output_png = "benchmarks/3d_lattice/field_comparison.png"
plt.savefig(output_png, dpi=150)
print(f"Saved benchmark timing plot to {output_png}")
