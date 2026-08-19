import time
import numpy as np
from cuMPM.dev import MonodisperseEwaldElectricField

def profile_run(num_particles=300000, num_repeats=10, xi=0.030):
    print("=" * 80, flush=True)
    print(f"PROFILING RUN: Monodisperse Ewald Field Solver (N = {num_particles:,}, xi = {xi})", flush=True)
    print("=" * 80, flush=True)

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

    # Warmup
    ewald.calculate()

    print("Phase 1: Running Synchronous evaluations...", flush=True)
    ewald.use_async_streams = False
    for _ in range(num_repeats):
        ewald.calculate()

    print("Phase 2: Running Asynchronous (Dual CUDA Streams) evaluations...", flush=True)
    ewald.use_async_streams = True
    for _ in range(num_repeats):
        ewald.calculate()

    print("Profiling iterations complete.", flush=True)

if __name__ == '__main__':
    profile_run()
