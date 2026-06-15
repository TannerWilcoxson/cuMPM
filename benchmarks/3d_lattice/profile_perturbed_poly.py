import time
import os
import numpy as np
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
omega = np.linspace(1000, 7000, 2) # 2 wavenumbers for profiling

# Two eps_p configurations with eps_inf = 4 for both
eps_p1 = drude_dielectric(omega, gamma=681, omega_p=12313, eps_inf=4)
eps_p2 = drude_dielectric(omega, gamma=400, omega_p=8000, eps_inf=4)

Nx, Ny, Nz = 80, 80, 40
N = Nx * Ny * Nz  # 256,000 particles

print(f"Setting up 3D perturbed lattice POLYDISPERSE profile with {N} particles (2 wavenumbers, 10 loops)...")

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

print("Warming up polydisperse solvers...")
# Warmup GMRES
solver_warmup_gmres = cuMPM.dipole_solver(
    box=warmup_box, eps_p=warmup_eps_p_mixed, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="gmres", field_type="polydisperse"
)
solver_warmup_gmres.compute(warmup_pos)

# Warmup BiCGSTAB
solver_warmup_bicg = cuMPM.dipole_solver(
    box=warmup_box, eps_p=warmup_eps_p_mixed, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="bicgstab", field_type="polydisperse"
)
solver_warmup_bicg.compute(warmup_pos)
print("Warmup complete.")

print(f"\nRunning solver execution for profiling...")
print("-" * 50)

# --- GMRES Profile ---
print("Running GMRES Solver (10x)...")
solver_gmres = cuMPM.dipole_solver(
    box=box, eps_p=eps_p_mixed, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="gmres", field_type="polydisperse"
)
t0 = time.time()
for run in range(10):
    t_start = time.time()
    solver_gmres.compute(pos)
    print(f"  GMRES run {run+1}/10: {time.time() - t_start:.4f} seconds")
t_gmres = time.time() - t0
print(f"GMRES execution time (10 runs): {t_gmres:.4f} seconds (average: {t_gmres/10.0:.4f}s)")

# --- BiCGSTAB Profile ---
print("Running BiCGSTAB Solver (10x)...")
solver_bicg = cuMPM.dipole_solver(
    box=box, eps_p=eps_p_mixed, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="bicgstab", field_type="polydisperse"
)
t0 = time.time()
for run in range(10):
    t_start = time.time()
    solver_bicg.compute(pos)
    print(f"  BiCGSTAB run {run+1}/10: {time.time() - t_start:.4f} seconds")
t_bicg = time.time() - t0
print(f"BiCGSTAB execution time (10 runs): {t_bicg:.4f} seconds (average: {t_bicg/10.0:.4f}s)")

print("-" * 50)
print(f"GMRES: {t_gmres:.4f}s total | BiCGSTAB: {t_bicg:.4f}s total")
