import numpy as np
import pytest
from cuMPM.dev import DirectElectricField, MonodisperseEwaldElectricField, PolydisperseEwaldElectricField

def test_direct_electric_field():
    radius = np.array([1.0, 1.2, 1.5])
    ef = DirectElectricField(radius, mode=False)
    
    x = np.array([0.0, 2.0, 4.0])
    y = np.array([0.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, 0.0])
    ef.update_particle_coordinates(x, y, z)
    ef.update_field_coordinates([1.0, 3.0], [0.0, 0.0], [0.0, 0.0])
    
    ef.update_dipoles_complex(
        np.ones(3), np.zeros(3),
        np.zeros(3), np.zeros(3),
        np.zeros(3), np.zeros(3)
    )
    ef.set_self_coef(np.ones(3), np.zeros(3))
    
    assert ef.num_particles == 3
    assert ef.num_field_points == 2
    
    ef.calculate()
    epoint = ef.get_epoint_host()
    assert epoint.shape == (2, 3)
    assert not np.any(np.isnan(epoint))

def test_ewald_equivalence_and_splits():
    box_x, box_y, box_z = 20.0, 20.0, 20.0
    errortol = 1e-6
    xi = 0.5
    calc_inter_dipole = True

    pos = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [15.0, 0.0, 0.0]
    ])
    num_particles = len(pos)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    # Non-zero dipoles and self-coefficients
    dip_x = np.array([1.0, 1.0, 1.0, 1.0])
    dip_y = np.zeros(num_particles)
    dip_z = np.zeros(num_particles)
    self_coef_r = np.ones(num_particles)
    self_coef_i = np.zeros(num_particles)

    # 1. Monodisperse Ewald
    ef_mono = MonodisperseEwaldElectricField(box_x, box_y, box_z, errortol, xi, calc_inter_dipole)
    ef_mono.update_particle_coordinates(x, y, z)
    ef_mono.update_dipoles(dip_x, dip_y, dip_z)
    ef_mono.set_self_coef(self_coef_r, self_coef_i)

    # 2. Polydisperse Ewald (initialized with radius = 1.0 for monodisperse equivalence)
    ef_poly = PolydisperseEwaldElectricField(box_x, box_y, box_z, errortol, xi, calc_inter_dipole, [1.0]*num_particles)
    ef_poly.update_particle_coordinates(x, y, z)
    ef_poly.update_dipoles(dip_x, dip_y, dip_z)
    ef_poly.set_self_coef(self_coef_r, self_coef_i)

    # Calculate fields
    ef_mono.calculate()
    E_tot_mono = ef_mono.get_epoint_host()

    ef_poly.calculate()
    E_tot_poly = ef_poly.get_epoint_host()

    # Assert mono vs poly total field equivalence
    # Due to different FFT grid size rules and spectral representations, tolerance is 1e-5
    np.testing.assert_allclose(E_tot_mono, E_tot_poly, rtol=1e-5, atol=1e-5)

    # Test split fields consistency: E_tot = E_real + E_inv
    E_real_mono = ef_mono.compute_real_field()
    E_inv_mono = ef_mono.compute_inverse_field()
    np.testing.assert_allclose(E_tot_mono, E_real_mono + E_inv_mono, rtol=1e-12, atol=1e-12)

    E_real_poly = ef_poly.compute_real_field()
    E_inv_poly = ef_poly.compute_inverse_field()
    np.testing.assert_allclose(E_tot_poly, E_real_poly + E_inv_poly, rtol=1e-12, atol=1e-12)

def test_ewald_quadrupoles_equivalence_and_splits():
    box_x, box_y, box_z = 20.0, 20.0, 20.0
    errortol = 1e-6
    xi = 0.5
    calc_inter_dipole = True

    pos = np.array([
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [12.0, 0.0, 0.0]
    ])
    num_particles = len(pos)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    dip_x = np.ones(num_particles)
    dip_y = np.zeros(num_particles)
    dip_z = np.zeros(num_particles)
    self_coef_r = np.ones(num_particles)
    self_coef_i = np.zeros(num_particles)

    # Quadrupoles setup
    quad_idxs = [0, 2]
    quad_1 = np.ones(len(quad_idxs))
    quad_2 = np.zeros(len(quad_idxs))
    quad_3 = np.zeros(len(quad_idxs))
    quad_4 = np.zeros(len(quad_idxs))
    quad_5 = np.zeros(len(quad_idxs))

    # 1. Monodisperse Ewald with Quadrupoles
    ef_mono = MonodisperseEwaldElectricField(
        box_x, box_y, box_z, errortol, xi, calc_inter_dipole,
        solve_quadrupoles=True, quad_idxs=quad_idxs
    )
    ef_mono.update_particle_coordinates(x, y, z)
    ef_mono.update_dipoles(dip_x, dip_y, dip_z)
    ef_mono.update_quadrupoles(quad_1, quad_2, quad_3, quad_4, quad_5)
    ef_mono.set_self_coef(self_coef_r, self_coef_i)

    # 2. Polydisperse Ewald with Quadrupoles (radius = 1.0)
    ef_poly = PolydisperseEwaldElectricField(
        box_x, box_y, box_z, errortol, xi, calc_inter_dipole, [1.0]*num_particles,
        solve_quadrupoles=True, quad_idxs=quad_idxs
    )
    ef_poly.update_particle_coordinates(x, y, z)
    ef_poly.update_dipoles(dip_x, dip_y, dip_z)
    ef_poly.update_quadrupoles(quad_1, quad_2, quad_3, quad_4, quad_5)
    ef_poly.set_self_coef(self_coef_r, self_coef_i)

    # Calculate fields
    ef_mono.calculate()
    E_tot_mono, G_tot_mono = ef_mono.get_epoint_host()

    ef_poly.calculate()
    E_tot_poly, G_tot_poly = ef_poly.get_epoint_host()

    # Compare E_tot (dipole field) and G_tot (quadrupole gradient field)
    # The real-space fields are identical, but minor reciprocal space differences exist due to different FFT/divergence grid formulations
    np.testing.assert_allclose(E_tot_mono, E_tot_poly, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(G_tot_mono, G_tot_poly, rtol=5e-3, atol=5e-3)


    # Test split fields consistency: E_tot = E_real + E_inv
    E_real_mono_dip, E_real_mono_quad = ef_mono.compute_real_field()
    E_inv_mono_dip, E_inv_mono_quad = ef_mono.compute_inverse_field()
    np.testing.assert_allclose(E_tot_mono, E_real_mono_dip + E_inv_mono_dip, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(G_tot_mono, E_real_mono_quad + E_inv_mono_quad, rtol=1e-12, atol=1e-12)

    E_real_poly_dip, E_real_poly_quad = ef_poly.compute_real_field()
    E_inv_poly_dip, E_inv_poly_quad = ef_poly.compute_inverse_field()
    np.testing.assert_allclose(E_tot_poly, E_real_poly_dip + E_inv_poly_dip, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(G_tot_poly, E_real_poly_quad + E_inv_poly_quad, rtol=1e-12, atol=1e-12)


def test_polydisperse_quadrupole_subset_electric_fields():
    """
    Directly compare the Monodisperse and Polydisperse Ewald electric field
    wrappers when a quadrupole subset (quad_idxs) is specified.
    """
    d = 10.0
    xi = 0.4
    tol = 1e-6
    radius = 1.0

    # 4 particles in a line
    pos = np.array([
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0],
        [2*d, 0.0, 0.0],
        [3*d, 0.0, 0.0]
    ])
    num_particles = len(pos)
    box = np.array([5*d, 5*d, 5*d])

    quad_idxs = [0, 2]
    solve_quadrupoles = True

    # Initialize field classes
    ef_mono = MonodisperseEwaldElectricField(
        box[0], box[1], box[2], tol, xi, True, radius=radius, solve_quadrupoles=solve_quadrupoles, quad_idxs=quad_idxs
    )
    ef_poly = PolydisperseEwaldElectricField(
        box[0], box[1], box[2], tol, xi, True, particle_radii=[radius] * num_particles, solve_quadrupoles=solve_quadrupoles, quad_idxs=quad_idxs
    )

    # Set coordinates
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    ef_mono.update_particle_coordinates(x, y, z)
    ef_poly.update_particle_coordinates(x, y, z)

    # Set self coefficients
    self_coef_r = np.ones(num_particles) * 1.5
    self_coef_i = np.ones(num_particles) * 0.1
    ef_mono.set_self_coef(self_coef_r, self_coef_i)
    ef_poly.set_self_coef(self_coef_r, self_coef_i)

    # Set dipoles & quadrupoles (random complex arrays)
    np.random.seed(42)
    P = np.random.rand(num_particles, 3) + 1j * np.random.rand(num_particles, 3)
    Q = np.random.rand(len(quad_idxs), 5) + 1j * np.random.rand(len(quad_idxs), 5)

    ef_mono.update_dipoles_complex(P[:, 0].real, P[:, 0].imag, P[:, 1].real, P[:, 1].imag, P[:, 2].real, P[:, 2].imag)
    ef_mono.update_quadrupoles_complex(Q[:, 0].real, Q[:, 0].imag, Q[:, 1].real, Q[:, 1].imag, Q[:, 2].real, Q[:, 2].imag, Q[:, 3].real, Q[:, 3].imag, Q[:, 4].real, Q[:, 4].imag)

    ef_poly.update_dipoles_complex(P[:, 0].real, P[:, 0].imag, P[:, 1].real, P[:, 1].imag, P[:, 2].real, P[:, 2].imag)
    ef_poly.update_quadrupoles_complex(Q[:, 0].real, Q[:, 0].imag, Q[:, 1].real, Q[:, 1].imag, Q[:, 2].real, Q[:, 2].imag, Q[:, 3].real, Q[:, 3].imag, Q[:, 4].real, Q[:, 4].imag)

    # Calculate
    ef_mono.calculate()
    ef_poly.calculate()

    # Get results
    E_mono, G_mono = ef_mono.get_epoint_host()
    E_poly, G_poly = ef_poly.get_epoint_host()

    # Assert matching
    np.testing.assert_allclose(E_mono, E_poly, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(G_mono, G_poly, rtol=1e-3, atol=1e-3)


def test_polydisperse_bloch_wavevector():
    """
    Test that the Polydisperse Ewald solver matches the Monodisperse Ewald solver
    for identical Bloch wavevector settings.
    """
    box_x, box_y, box_z = 18.0, 18.0, 18.0
    errortol = 1e-6
    xi = 0.5
    calc_inter_dipole = True
    quad_idxs = [0, 2]

    pos = np.array([
        [0.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
        [9.0, 0.0, 0.0],
        [13.5, 0.0, 0.0]
    ])
    num_particles = len(pos)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    # Initialize solvers
    ef_mono = MonodisperseEwaldElectricField(box_x, box_y, box_z, errortol, xi, calc_inter_dipole, solve_quadrupoles=True, quad_idxs=quad_idxs)
    ef_poly = PolydisperseEwaldElectricField(box_x, box_y, box_z, errortol, xi, calc_inter_dipole, [1.0]*num_particles, solve_quadrupoles=True, quad_idxs=quad_idxs)

    # Set Bloch wavevectors
    kx, ky = 0.15, -0.22
    ef_mono.set_bloch_wavevector(kx, ky)
    ef_poly.set_bloch_wavevector(kx, ky)

    ef_mono.update_particle_coordinates(x, y, z)
    ef_poly.update_particle_coordinates(x, y, z)

    # Set same complex self coefficients, dipoles, and quadrupoles
    self_coef_r = np.ones(num_particles)
    self_coef_i = np.zeros(num_particles)
    ef_mono.set_self_coef(self_coef_r, self_coef_i)
    ef_poly.set_self_coef(self_coef_r, self_coef_i)

    np.random.seed(123)
    P = np.random.rand(num_particles, 3) + 1j * np.random.rand(num_particles, 3)
    Q = np.random.rand(len(quad_idxs), 5) + 1j * np.random.rand(len(quad_idxs), 5)

    ef_mono.update_dipoles_complex(P[:, 0].real, P[:, 0].imag, P[:, 1].real, P[:, 1].imag, P[:, 2].real, P[:, 2].imag)
    ef_mono.update_quadrupoles_complex(Q[:, 0].real, Q[:, 0].imag, Q[:, 1].real, Q[:, 1].imag, Q[:, 2].real, Q[:, 2].imag, Q[:, 3].real, Q[:, 3].imag, Q[:, 4].real, Q[:, 4].imag)

    ef_poly.update_dipoles_complex(P[:, 0].real, P[:, 0].imag, P[:, 1].real, P[:, 1].imag, P[:, 2].real, P[:, 2].imag)
    ef_poly.update_quadrupoles_complex(Q[:, 0].real, Q[:, 0].imag, Q[:, 1].real, Q[:, 1].imag, Q[:, 2].real, Q[:, 2].imag, Q[:, 3].real, Q[:, 3].imag, Q[:, 4].real, Q[:, 4].imag)

    # Calculate
    ef_mono.calculate()
    ef_poly.calculate()

    # Get results
    E_mono, G_mono = ef_mono.get_epoint_host()
    E_poly, G_poly = ef_poly.get_epoint_host()

    # Assert matching
    np.testing.assert_allclose(E_mono, E_poly, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(G_mono, G_poly, rtol=1e-3, atol=1e-3)


def test_direct_electric_field_precision_modes():
    from cuMPM._cuMPM import PrecisionMode

    radius = np.array([1.0, 1.2, 1.5])

    # 1. Explicit FP64 Direct Field
    ef_64 = DirectElectricField(radius, mode=False, precision=PrecisionMode.FP64)
    assert not ef_64.is_using_fp32
    assert ef_64.precision_mode == PrecisionMode.FP64

    # 2. Explicit FP32 Direct Field
    ef_32 = DirectElectricField(radius, mode=False, precision=PrecisionMode.FP32)
    assert ef_32.is_using_fp32
    assert ef_32.precision_mode == PrecisionMode.FP32

    x = np.array([0.0, 2.0, 4.0])
    y = np.array([0.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, 0.0])

    ef_64.update_particle_coordinates(x, y, z)
    ef_64.update_field_coordinates([1.0, 3.0], [0.0, 0.0], [0.0, 0.0])
    ef_64.update_dipoles_complex(np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    ef_64.set_self_coef(np.ones(3), np.zeros(3))
    ef_64.calculate()
    epoint_64 = ef_64.get_epoint_host()

    ef_32.update_particle_coordinates(x, y, z)
    ef_32.update_field_coordinates([1.0, 3.0], [0.0, 0.0], [0.0, 0.0])
    ef_32.update_dipoles_complex(np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    ef_32.set_self_coef(np.ones(3), np.zeros(3))
    ef_32.calculate()
    epoint_32 = ef_32.get_epoint_host()

    # FP32 and FP64 results should match to single-precision float tolerance (~1e-5)
    np.testing.assert_allclose(epoint_32, epoint_64, rtol=1e-5, atol=1e-5)


def test_ewald_electric_field_precision_modes():
    from cuMPM._cuMPM import PrecisionMode

    box = (50.0, 50.0, 50.0)
    errortol = 1e-4
    xi = 0.5
    radius = 1.0

    # 1. Monodisperse Ewald in DOUBLE mode vs MIXED mode
    ef_mono_double = MonodisperseEwaldElectricField(*box, errortol, xi, calc_inter_dipole=True, radius=radius, precision="double")
    assert not ef_mono_double.is_using_recip_fp32
    assert ef_mono_double.recip_precision_mode == PrecisionMode.DOUBLE

    ef_mono_mixed = MonodisperseEwaldElectricField(*box, errortol, xi, calc_inter_dipole=True, radius=radius, precision="mixed")
    assert ef_mono_mixed.is_using_recip_fp32
    assert ef_mono_mixed.recip_precision_mode == PrecisionMode.MIXED

    x = np.array([0.0, 2.0, 4.0])
    y = np.array([0.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, 0.0])
    dip_x = np.ones(3)

    ef_mono_double.update_particle_coordinates(x, y, z)
    ef_mono_double.update_dipoles(dip_x, np.zeros(3), np.zeros(3))
    ef_mono_double.calculate()
    epoint_double = ef_mono_double.get_epoint_host()

    ef_mono_mixed.update_particle_coordinates(x, y, z)
    ef_mono_mixed.update_dipoles(dip_x, np.zeros(3), np.zeros(3))
    ef_mono_mixed.calculate()
    epoint_mixed = ef_mono_mixed.get_epoint_host()

    # Reciprocal FP32 and FP64 results should match within single precision tolerance (~1e-4)
    np.testing.assert_allclose(epoint_mixed, epoint_double, rtol=1e-4, atol=1e-4)




