import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf=4):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

def make_perturbed_cubic_grid(Nx, Ny, Nz, spacing):
    """Generate 3D simple cubic grid coordinates perturbed within their unit cells."""
    positions = []
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                positions.append([i * spacing, j * spacing, k * spacing])
    pos = np.array(positions)
    # Random perturbation within unit cell [-spacing/2, spacing/2]
    np.random.seed(42)  # For reproducibility
    perturbations = np.random.uniform(-spacing/2, spacing/2, size=pos.shape)
    return pos + perturbations

# Parameters
spacing = 1.2
radius = 0.5
eps_m = 2.13
omega = np.linspace(1000, 7000, 80) # 80 wavenumbers

# Two eps_p configurations with eps_inf = 4 for both
eps_p1 = drude_dielectric(omega, gamma=681, omega_p=12313, eps_inf=4)
eps_p2 = drude_dielectric(omega, gamma=400, omega_p=8000, eps_inf=4)

Nx, Ny, Nz = 20, 20, 20
N = Nx * Ny * Nz  # 8,000 particles

print(f"Setting up 3D perturbed lattice benchmark with {N} particles...")

# Generate perturbed coordinates
pos = make_perturbed_cubic_grid(Nx, Ny, Nz, spacing)

# Calculate box sizes based on lattice period
box = np.array([
    Nx * spacing,
    Ny * spacing,
    Nz * spacing
])

# Randomly assign each particle to one of the two permittivities
np.random.seed(42)  # For reproducibility
choices = np.random.randint(0, 2, size=N)
eps_p_mixed = np.where(choices[None, :] == 0, eps_p1[:, None], eps_p2[:, None])

# Warmup run to initialize CUDA context, CUFFT plans, etc.
warmup_N = 1000
warmup_Nx, warmup_Ny, warmup_Nz = 10, 10, 10
warmup_pos = make_perturbed_cubic_grid(warmup_Nx, warmup_Ny, warmup_Nz, spacing)
warmup_box = np.array([warmup_Nx * spacing, warmup_Ny * spacing, warmup_Nz * spacing])

np.random.seed(123)
warmup_choices = np.random.randint(0, 2, size=warmup_N)
warmup_eps_p_mixed = np.where(warmup_choices[None, :] == 0, eps_p1[:, None], eps_p2[:, None])

print("Warming up solvers...")
# Warmup GMRES
solver_warmup_gmres = cuMPM.dipole_solver(
    box=warmup_box, eps_p=warmup_eps_p_mixed, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="gmres"
)
solver_warmup_gmres.compute(warmup_pos)

# Warmup BiCGSTAB
solver_warmup_bicg = cuMPM.dipole_solver(
    box=warmup_box, eps_p=warmup_eps_p_mixed, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="bicgstab"
)
solver_warmup_bicg.compute(warmup_pos)
print("Warmup complete.")

print(f"\nRunning solver benchmark for {N} particles (80 wavenumbers)...")
print("-" * 50)

# --- Benchmark GMRES ---
print("Running GMRES Solver...")
solver_gmres = cuMPM.dipole_solver(
    box=box, eps_p=eps_p_mixed, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="gmres"
)
t0 = time.time()
solver_gmres.compute(pos)
t_gmres = time.time() - t0
print(f"GMRES execution time: {t_gmres:.4f} seconds")

# --- Benchmark BiCGSTAB ---
print("Running BiCGSTAB Solver...")
solver_bicg = cuMPM.dipole_solver(
    box=box, eps_p=eps_p_mixed, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="bicgstab"
)
t0 = time.time()
solver_bicg.compute(pos)
t_bicg = time.time() - t0
print(f"BiCGSTAB execution time: {t_bicg:.4f} seconds")

print("-" * 50)
print(f"GMRES: {t_gmres:.4f}s | BiCGSTAB: {t_bicg:.4f}s")
print(f"BiCGSTAB is {t_gmres / t_bicg:.2f}x faster than GMRES" if t_gmres > t_bicg else f"GMRES is {t_bicg / t_gmres:.2f}x faster than BiCGSTAB")

# Save a simple bar plot
solvers = ['GMRES', 'BiCGSTAB']
times = [t_gmres, t_bicg]

plt.figure(figsize=(6, 5))
colors = ['#1f77b4', '#d62728']
bars = plt.bar(solvers, times, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
plt.ylabel('Execution Time (seconds)', fontsize=11)
plt.title('cuMPM Solver Timing Comparison\n(8k perturbed particles, 80 wavenumbers)', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle=':', alpha=0.5)

# Add text labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02 * max(times), f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Save plot
os.makedirs("benchmarks/3d_lattice", exist_ok=True)
output_png = "benchmarks/3d_lattice/perturbed_solver_comparison.png"
plt.savefig(output_png, dpi=150)
print(f"Saved benchmark timing plot to {output_png}")
