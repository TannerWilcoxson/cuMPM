"""
plot_split_dipoles.py
Visualise the spatial distribution and orientations of the split sub-dipoles
produced by the cuMPM EELS splitting algorithm at a single impact parameter
and at the ITO LSPR peak wavenumber.

Panels
------
1. Top-left  – 3-D scatter of sub-dipole positions, coloured by |p| at peak ω
2. Top-right – XY cross-section with quiver arrows showing Re(p_x), Re(p_y)
3. Bottom-left  – |p| vs wavenumber for every sub-dipole (translucent) + mean
4. Bottom-right – Im(p_z) vs Re(p_z) phasor plot at peak ω
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import cuMPM

# ── System parameters (match eels_single_ito_split.py) ─────────────────────
box      = [100.0, 100.0, 100.0]
radius   = 5.0          # nm
eps_m    = 1.0
v        = 0.5          # 80 keV  (β = 0.5c)
N_split  = 400
epos     = np.array([[4.67, 0.0]])   # nm, single impact parameter inside NC
pos      = np.array([[0.0, 0.0, 0.0]])

omega_cm  = np.linspace(3000, 9000, 120)   # cm⁻¹
omega_nm  = omega_cm / 1e7                  # nm⁻¹

# ITO Drude parameters
omega_p  = 11886.0
gamma    = 845.0
eps_inf  = 2.25
eps_p    = eps_inf - omega_p**2 / (omega_cm**2 + 1j * omega_cm * gamma)

# ── Solver ──────────────────────────────────────────────────────────────────
split_dist = 1.2 * radius
solver = cuMPM.EELS(
    box=box, eps_p=eps_p, omega=omega_nm, v=v,
    eps_m=eps_m, radius=radius,
    split_dist=split_dist, N_split=N_split,
    field_type="auto", quiet=True
)

print(f"Computing split EELS (N_split={N_split}, epos={epos[0]} nm)…")
t0 = time.time()
solver.compute(epos, pos)
print(f"  Done in {time.time()-t0:.1f}s")

# ── Extract results ──────────────────────────────────────────────────────────
# dips shape: (num_wavevectors, num_split_particles, 3) – complex
dips    = solver.get_dipoles()          # (120, N_split, 3)
spos    = solver.get_positions()        # (N_split, 3)  sub-dipole centres

num_omega, num_sub, _ = dips.shape

# Magnitude |p| = sqrt(|p_x|^2 + |p_y|^2 + |p_z|^2) at each ω
pmag = np.sqrt(np.sum(np.abs(dips)**2, axis=-1))   # (120, N_split)

# Peak ω index — take the wavenumber of maximum mean |p|
peak_idx = int(np.argmax(pmag.mean(axis=1)))
peak_omega = omega_cm[peak_idx]
print(f"  Peak |p| wavenumber: {peak_omega:.0f} cm⁻¹  (index {peak_idx})")

p_peak = dips[peak_idx]          # (N_split, 3) complex at peak ω
mag_peak = pmag[peak_idx]        # (N_split,)   magnitude at peak ω

# ── Plotting ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
fig.suptitle(
    f"Split Dipoles — ITO NC  $R={radius}$ nm,  "
    f"$N_{{\\mathrm{{split}}}}={N_split}$,  "
    f"$x_{{\\mathrm{{ebeam}}}}={epos[0,0]}$ nm",
    fontsize=13, fontweight='bold'
)

cmap = cm.plasma
norm_peak = plt.Normalize(mag_peak.min(), mag_peak.max())
colors_peak = cmap(norm_peak(mag_peak))

# ── Panel 1: 3-D positions coloured by |p| ──────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
sc = ax1.scatter(
    spos[:, 0], spos[:, 1], spos[:, 2],
    c=mag_peak, cmap=cmap, s=18, alpha=0.85, edgecolors='none'
)
# Draw e-beam trajectory (dashed line along z at epos x,y)
z_range = np.linspace(-radius * 1.5, radius * 1.5, 50)
ax1.plot(
    np.full_like(z_range, epos[0, 0]),
    np.full_like(z_range, epos[0, 1]),
    z_range,
    'c--', linewidth=1.5, label='e-beam', alpha=0.8
)
# Draw original NC sphere as a wireframe
u = np.linspace(0, 2 * np.pi, 30)
vv = np.linspace(0, np.pi, 20)
xs = radius * np.outer(np.cos(u), np.sin(vv))
ys = radius * np.outer(np.sin(u), np.sin(vv))
zs = radius * np.outer(np.ones_like(u), np.cos(vv))
ax1.plot_wireframe(xs, ys, zs, color='gray', alpha=0.12, linewidth=0.5)

fig.colorbar(sc, ax=ax1, shrink=0.6, label='$|p|$ (arb.)')
ax1.set_xlabel('x (nm)'); ax1.set_ylabel('y (nm)'); ax1.set_zlabel('z (nm)')
ax1.set_title(f'Sub-dipole positions\n$|p|$ at $\\omega={peak_omega:.0f}$ cm$^{{-1}}$',
              fontsize=10)
ax1.legend(fontsize=8, loc='upper left')

# ── Panel 2: XY cross-section with quiver arrows ─────────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
# Background colour = |p|
sc2 = ax2.scatter(
    spos[:, 0], spos[:, 1],
    c=mag_peak, cmap=cmap, s=30, alpha=0.9, edgecolors='none', zorder=3
)
# Quiver: Re(p_x), Re(p_y)
px_re = p_peak[:, 0].real
py_re = p_peak[:, 1].real
qscale = mag_peak.max() * 0.8 / (np.sqrt(px_re**2 + py_re**2).max() + 1e-30)
ax2.quiver(
    spos[:, 0], spos[:, 1], px_re * qscale, py_re * qscale,
    color='white', scale=1, scale_units='xy', angles='xy',
    width=0.003, alpha=0.7, zorder=4
)
# NC circle
theta = np.linspace(0, 2 * np.pi, 200)
ax2.plot(radius * np.cos(theta), radius * np.sin(theta),
         'w--', linewidth=1.5, label=f'NC boundary R={radius} nm', zorder=5)
# E-beam position
ax2.axvline(epos[0, 0], color='cyan', linestyle='--', linewidth=1.5,
            label=f'e-beam ($x={epos[0,0]}$ nm)', zorder=5)
fig.colorbar(sc2, ax=ax2, label='$|p|$ (arb.)')
ax2.set_xlabel('x (nm)'); ax2.set_ylabel('y (nm)')
ax2.set_aspect('equal')
ax2.set_title('XY cross-section — Re($p_x$, $p_y$) arrows', fontsize=10)
ax2.legend(fontsize=8)

# ── Panel 3: |p| spectrum per sub-dipole ────────────────────────────────────
ax3 = fig.add_subplot(2, 2, 3)
# Colour sub-dipoles by their distance from the e-beam impact point
dist_to_beam = np.sqrt((spos[:, 0] - epos[0, 0])**2 + spos[:, 1]**2)
norm_dist = plt.Normalize(dist_to_beam.min(), dist_to_beam.max())
for j in range(num_sub):
    ax3.plot(
        omega_cm, pmag[:, j],
        color=cm.coolwarm(norm_dist(dist_to_beam[j])),
        alpha=0.25, linewidth=0.6
    )
ax3.plot(omega_cm, pmag.mean(axis=1), 'k-', linewidth=2, label='Mean $|p|$')
ax3.axvline(peak_omega, color='gray', linestyle=':', linewidth=1)
ax3.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=10)
ax3.set_ylabel('$|p|$ (arb.)', fontsize=10)
ax3.set_title('Dipole magnitude spectrum\n(colour = distance to e-beam)', fontsize=10)
sm = cm.ScalarMappable(cmap='coolwarm', norm=norm_dist)
sm.set_array([])
fig.colorbar(sm, ax=ax3, label='Dist. to beam (nm)')
ax3.legend(fontsize=9)

# ── Panel 4: Phasor plot Im(p_z) vs Re(p_z) at peak ω ───────────────────────
ax4 = fig.add_subplot(2, 2, 4)
pz_peak = p_peak[:, 2]
sc4 = ax4.scatter(
    pz_peak.real, pz_peak.imag,
    c=dist_to_beam, cmap='coolwarm', s=22, alpha=0.85, edgecolors='none'
)
ax4.axhline(0, color='gray', linewidth=0.6)
ax4.axvline(0, color='gray', linewidth=0.6)
ax4.set_xlabel('Re($p_z$) (arb.)', fontsize=10)
ax4.set_ylabel('Im($p_z$) (arb.)', fontsize=10)
ax4.set_title(f'$p_z$ phasor at $\\omega={peak_omega:.0f}$ cm$^{{-1}}$\n'
              f'(colour = distance to e-beam)', fontsize=10)
ax4.set_aspect('equal')
fig.colorbar(sc4, ax=ax4, label='Dist. to beam (nm)')

plt.tight_layout()
outfile = 'split_dipoles.png'
plt.savefig(outfile, dpi=200)
plt.show()
print(f"Saved → {outfile}")
