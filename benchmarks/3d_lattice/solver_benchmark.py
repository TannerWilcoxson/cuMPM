import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

def drude_dielectric(omega, gamma=681, omega_p=12313, eps_inf=4):
    """Calculate the Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

def make_simple_cubic_grid(N, spacing=12.0):
    """Generate 3D simple cubic grid coordinates for N particles."""
    positions = []
    limit = int(np.ceil(N ** (1/3))) + 1
    for k in range(limit):
        for j in range(limit):
            for i in range(limit):
                positions.append([i * spacing, j * spacing, k * spacing])
                if len(positions) == N:
                    break
            if len(positions) == N:
                break
        if len(positions) == N:
            break
    return np.array(positions)

# Parameters
spacing = 12.0
radius = 5.0
eps_m = 2.13
omega = np.linspace(1000, 7000, 80) # 80 wavenumbers as requested
eps_p = drude_dielectric(omega)

particle_counts = [10000, 50000, 100000, 500000]
gmres_times = []
bicgstab_times = []

print(f"Starting 3D Solver Benchmark (80 wavenumbers)...")
print(f"{'N':<8}{'GMRES (s)':<15}{'BiCGSTAB (s)':<15}")
print("-" * 40)

# Warmup run to initialize CUDA context, CUFFT plans, etc.
warmup_pos = make_simple_cubic_grid(1000, spacing)
rc = 5.256522
min_box_len = 2.1 * rc * radius
warmup_max_x = np.max(warmup_pos[:, 0])
warmup_max_y = np.max(warmup_pos[:, 1])
warmup_max_z = np.max(warmup_pos[:, 2])
warmup_box = np.array([
    max(warmup_max_x + spacing, min_box_len),
    max(warmup_max_y + spacing, min_box_len),
    max(warmup_max_z + spacing, min_box_len)
])
warmup_solver = cuMPM.dipole_solver(box=warmup_box, eps_p=eps_p, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True)
warmup_solver.compute(warmup_pos)

for N in particle_counts:
    # Generate 3D grid
    pos = make_simple_cubic_grid(N, spacing)
    
    # Calculate box sizes (ensuring box is at least 2.1 * rc * radius)
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 1])
    max_z = np.max(pos[:, 2])
    box = np.array([
        max(max_x + spacing, min_box_len),
        max(max_y + spacing, min_box_len),
        max(max_z + spacing, min_box_len)
    ])
    
    # --- Benchmark GMRES ---
    solver_gmres = cuMPM.dipole_solver(
        box=box, eps_p=eps_p, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="gmres"
    )
    t0 = time.time()
    solver_gmres.compute(pos)
    t_gmres = time.time() - t0
    gmres_times.append(t_gmres)
    
    # --- Benchmark BiCGSTAB ---
    solver_bicg = cuMPM.dipole_solver(
        box=box, eps_p=eps_p, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="bicgstab"
    )
    t0 = time.time()
    solver_bicg.compute(pos)
    t_bicg = time.time() - t0
    bicgstab_times.append(t_bicg)
    
    print(f"{N:<8}{t_gmres:<15.4f}{t_bicg:<15.4f}")

# Plotting
plt.figure(figsize=(8, 6))
plt.loglog(particle_counts, gmres_times, 'o-', color='blue', linewidth=2, markersize=8, label="GMRES Solver")
plt.loglog(particle_counts, bicgstab_times, 's--', color='red', linewidth=2, markersize=8, label="BiCGSTAB Solver")

plt.title("cuMPM 3D Solver Runtime Scaling (80 Wavenumbers)", fontsize=13, fontweight='bold')
plt.xlabel("Number of Particles ($N$)", fontsize=11)
plt.ylabel("Execution Time (seconds)", fontsize=11)
plt.grid(True, which="both", linestyle=":", alpha=0.5)
plt.legend(frameon=True, shadow=True, fontsize=11)
plt.tight_layout()

# Save plot
os.makedirs("benchmarks/3d_lattice", exist_ok=True)
output_png = "benchmarks/3d_lattice/solver_comparison.png"
plt.savefig(output_png, dpi=150)
print(f"Saved benchmark timing plot to {output_png}")
