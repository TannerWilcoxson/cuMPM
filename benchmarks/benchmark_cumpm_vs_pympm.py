import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cuMPM

try:
    import pyMPM
    HAS_PYMPM = True
except ImportError:
    HAS_PYMPM = False
    print("Warning: pyMPM is not installed. pyMPM comparison will be skipped.")

def make_hexagonal_lattice(N, spacing=12.0):
    """Generate 2D hexagonal lattice coordinates for N particles."""
    positions = []
    limit = int(np.ceil(np.sqrt(N))) * 2
    for j in range(limit):
        for i in range(limit):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * (np.sqrt(3) / 2) * spacing
            positions.append([x, y, 0.0])
            if len(positions) == N:
                break
        if len(positions) == N:
            break
    return np.array(positions)

def drude_dielectric(omega, gamma=681, omega_p=12313, eps_inf=4):
    """Calculate Drude dielectric function."""
    return eps_inf - omega_p**2 / (omega**2 + 1j * omega * gamma)

def run_benchmark():
    # 1. Benchmark Parameters
    N = 4000
    spacing = 12.0
    radius = 5.0
    eps_m = 2.13
    
    # 10 frequency points for direct benchmark comparison
    num_freqs = 10
    omega = np.linspace(2000, 6000, num_freqs)
    eps_p = drude_dielectric(omega)
    
    print("=" * 65)
    print(f"  cuMPM vs. pyMPM Performance Benchmark (N = {N} particles)")
    print("=" * 65)
    
    # Generate 4000-particle lattice
    pos = make_hexagonal_lattice(N, spacing)
    rc = 5.256522
    min_box_len = 2.1 * rc * radius
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 1])
    box = np.array([
        max(max_x + spacing, min_box_len),
        max(max_y + spacing * np.sqrt(3)/2, min_box_len),
        1000.0
    ])
    
    print(f"System Configuration:")
    print(f"  Particles (N):        {N}")
    print(f"  Box Dimensions (nm):  {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f}")
    print(f"  Frequency Points:     {num_freqs}")
    print("-" * 65)
    
    # 2. Warmup cuMPM (GPU Context / CUFFT plan allocation)
    print("Warming up CUDA GPU solver context...")
    warmup_pos = pos[:100]
    warmup_box = np.array([box[0], box[1], box[2]])
    warmup_solver = cuMPM.dipole_solver(box=warmup_box, eps_p=eps_p[:2], radius=radius, eps_m=eps_m, tol=1e-3, quiet=True)
    warmup_solver.compute(warmup_pos)
    
    # 3. cuMPM Benchmark (BiCGSTAB & GMRES)
    print("\n[1/3] Running cuMPM (CUDA GPU - BiCGSTAB)...")
    cu_bicg_solver = cuMPM.dipole_solver(box=box, eps_p=eps_p, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="bicgstab")
    t0 = time.time()
    cu_bicg_solver.compute(pos)
    t_cu_bicg = time.time() - t0
    alpha_cu_bicg = cu_bicg_solver.get_eff_polarizability()
    print(f"      cuMPM (BiCGSTAB) completed in: {t_cu_bicg:.4f} seconds ({t_cu_bicg/num_freqs*1000:.2f} ms/freq)")

    print("\n[2/3] Running cuMPM (CUDA GPU - GMRES)...")
    cu_gmres_solver = cuMPM.dipole_solver(box=box, eps_p=eps_p, radius=radius, eps_m=eps_m, tol=1e-3, quiet=True, solver_type="gmres")
    t0 = time.time()
    cu_gmres_solver.compute(pos)
    t_cu_gmres = time.time() - t0
    alpha_cu_gmres = cu_gmres_solver.get_eff_polarizability()
    print(f"      cuMPM (GMRES) completed in:    {t_cu_gmres:.4f} seconds ({t_cu_gmres/num_freqs*1000:.2f} ms/freq)")

    # 4. pyMPM Benchmark (CPU Python)
    t_py = None
    speedup_bicg = None
    speedup_gmres = None
    max_diff = None

    if HAS_PYMPM:
        print("\n[3/3] Running pyMPM (CPU Python implementation)...")
        mpm_py = pyMPM.MPM(box, eps_p, radius=radius, eps_m=eps_m)
        t0 = time.time()
        mpm_py.compute(pos)
        t_py = time.time() - t0
        alpha_py = mpm_py.get_eff_polarizability()
        print(f"      pyMPM completed in:           {t_py:.4f} seconds ({t_py/num_freqs*1000:.2f} ms/freq)")

        # Verify numerical equivalence
        diff = np.abs(alpha_py - alpha_cu_bicg)
        max_diff = np.max(diff)
        speedup_bicg = t_py / t_cu_bicg
        speedup_gmres = t_py / t_cu_gmres
    else:
        print("\n[3/3] pyMPM skipping...")

    # 5. Summary Results
    print("\n" + "=" * 65)
    print("  BENCHMARK SUMMARY RESULTS (N = 4,000)")
    print("=" * 65)
    print(f"{'Implementation':<22}{'Total Time (s)':<18}{'Time/Freq (ms)':<18}{'Speedup':<10}")
    print("-" * 65)
    if HAS_PYMPM:
        print(f"{'pyMPM (CPU)':<22}{t_py:<18.4f}{t_py/num_freqs*1000:<18.2f}{'1.0x':<10}")
    print(f"{'cuMPM (GPU - BiCGSTAB)':<22}{t_cu_bicg:<18.4f}{t_cu_bicg/num_freqs*1000:<18.2f}{f'{speedup_bicg:.1f}x' if speedup_bicg else 'N/A':<10}")
    print(f"{'cuMPM (GPU - GMRES)':<22}{t_cu_gmres:<18.4f}{t_cu_gmres/num_freqs*1000:<18.2f}{f'{speedup_gmres:.1f}x' if speedup_gmres else 'N/A':<10}")
    print("-" * 65)

    if max_diff is not None:
        print(f"Max Absolute Error (cuMPM vs pyMPM): {max_diff:.3e}")
        print(f"Numerical Equivalence Status: {'PASSED (Match < 1e-4)' if max_diff < 1e-4 else 'WARNING (Diff > 1e-4)'}")

    # 6. Visualization Chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    solvers = ['pyMPM\n(CPU)', 'cuMPM\n(GPU - BiCGSTAB)', 'cuMPM\n(GPU - GMRES)'] if HAS_PYMPM else ['cuMPM (BiCGSTAB)', 'cuMPM (GMRES)']
    times = [t_py, t_cu_bicg, t_cu_gmres] if HAS_PYMPM else [t_cu_bicg, t_cu_gmres]
    bar_colors = ['#e76f51', '#2a9d8f', '#264653'] if HAS_PYMPM else ['#2a9d8f', '#264653']

    bars = ax.bar(solvers, times, color=bar_colors, width=0.55, edgecolor='black', linewidth=0.8)

    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_title(f"cuMPM vs. pyMPM Benchmark Runtime (N = {N} Particles, {num_freqs} Wavenumbers)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # Annotate values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f} s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    if speedup_bicg:
        ax.annotate(f'{speedup_bicg:.1f}x Speedup',
                    xy=(bars[1].get_x() + bars[1].get_width() / 2, bars[1].get_height()),
                    xytext=(0, 22),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2a9d8f')

    plt.tight_layout()
    output_png = "benchmarks/benchmark_cumpm_vs_pympm_4000.png"
    plt.savefig(output_png, dpi=200)
    print(f"\nSaved benchmark plot to {output_png}")

if __name__ == "__main__":
    run_benchmark()
