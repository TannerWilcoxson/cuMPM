import time
import numpy as np
from cuMPM.dev import MonodisperseEwaldElectricField

def run_benchmark(num_particles=30000, num_repeats=30, xi=0.20):
    print("=" * 80, flush=True)
    print(f"BENCHMARK: Monodisperse Ewald Field Solver (N = {num_particles:,}, xi = {xi})", flush=True)
    print("=" * 80, flush=True)

    # 3D Grid placement with ZERO particle overlap (minimum spacing d = 15 nm)
    target_spacing = 15.0
    n_side = int(np.ceil(num_particles ** (1.0 / 3.0)))
    box_size = float(np.round(n_side * target_spacing, 1))
    box = [box_size, box_size, box_size]
    errortol = 1e-4

    grid_coords = np.linspace(-box_size / 2.2, box_size / 2.2, n_side)
    gx, gy, gz = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
    pos_x = gx.flatten()[:num_particles]
    pos_y = gy.flatten()[:num_particles]
    pos_z = gz.flatten()[:num_particles]

    np.random.seed(42)
    dip_xr = np.random.randn(num_particles)
    dip_xi = np.random.randn(num_particles)
    dip_yr = np.random.randn(num_particles)
    dip_yi = np.random.randn(num_particles)
    dip_zr = np.random.randn(num_particles)
    dip_zi = np.random.randn(num_particles)

    self_r = np.ones(num_particles)
    self_i = np.zeros(num_particles)

    ewald = MonodisperseEwaldElectricField(
        box_x=box[0], box_y=box[1], box_z=box[2],
        errortol=errortol, xi=xi,
        calc_inter_dipole=False,
        solve_quadrupoles=False,
        precision="mixed"
    )

    ewald.update_particle_coordinates(pos_x, pos_y, pos_z)
    ewald.update_field_coordinates(pos_x, pos_y, pos_z)
    ewald.update_dipoles_complex(dip_xr, dip_xi, dip_yr, dip_yi, dip_zr, dip_zi)
    ewald.set_self_coef(self_r, self_i)

    # 1. Benchmark Synchronous Route (Single Default Stream)
    ewald.use_async_streams = False
    for _ in range(3):
        ewald.calculate()

    t0 = time.perf_counter()
    for _ in range(num_repeats):
        ewald.calculate()
    t_sync = (time.perf_counter() - t0) / num_repeats * 1000.0
    out_sync = ewald.get_epoint_host()

    # 2. Benchmark Asynchronous Route (Dual CUDA Streams)
    ewald.use_async_streams = True
    for _ in range(3):
        ewald.calculate()

    t0 = time.perf_counter()
    for _ in range(num_repeats):
        ewald.calculate()
    t_async = (time.perf_counter() - t0) / num_repeats * 1000.0
    out_async = ewald.get_epoint_host()

    diff = np.max(np.abs(np.array(out_sync) - np.array(out_async)))
    speedup = ((t_sync - t_async) / t_sync) * 100.0
    ratio = t_sync / t_async

    rc = np.sqrt(-np.log(errortol)) / xi
    print(f"  Box Size                          : {box_size:.1f} x {box_size:.1f} x {box_size:.1f} nm", flush=True)
    print(f"  Real-Space Cutoff (rc)            : {rc:.2f} nm", flush=True)
    print(f"  Synchronous Time (Single Stream)  : {t_sync:8.3f} ms / eval", flush=True)
    print(f"  Asynchronous Time (Dual Streams)   : {t_async:8.3f} ms / eval", flush=True)
    print(f"  Speedup                            : {speedup:8.2f} %  ({ratio:.2f}x faster)", flush=True)
    print(f"  Max Absolute Field Difference      : {diff:8.2e}", flush=True)
    print(flush=True)

    return {
        'num_particles': num_particles,
        'box_size': box_size,
        'rc': rc,
        't_sync_ms': t_sync,
        't_async_ms': t_async,
        'speedup_pct': speedup,
        'ratio': ratio,
        'max_diff': diff
    }

def main():
    print("=" * 80, flush=True)
    print("      cuMPM EWALD FIELD SOLVER BENCHMARK (xi = 0.20 nm^-1)", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    sizes = [30000, 100000, 300000]
    results = []

    for N in sizes:
        res = run_benchmark(num_particles=N, num_repeats=30, xi=0.20)
        results.append(res)

    print("=" * 80, flush=True)
    print("                      SUMMARY BENCHMARK RESULTS TABLE (xi = 0.20)")
    print("=" * 80, flush=True)
    print(f"{'Particles (N)':<15} | {'Cutoff rc (nm)':<15} | {'Sync Time (ms)':<16} | {'Async Time (ms)':<16} | {'Speedup (%)':<12} | {'Ratio':<8}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        print(f"{r['num_particles']:<15,} | {r['rc']:<15.2f} | {r['t_sync_ms']:<16.3f} | {r['t_async_ms']:<16.3f} | {r['speedup_pct']:<12.2f}% | {r['ratio']:<8.2f}x", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
