import numpy as np
from . import _cuMPM

def drude_dielectric(omega, gamma, omega_p, eps_inf):
    """
    Calculate the complex relative permittivity of a Drude metal.

    Parameters
    ----------
    omega : float or array_like
        Wavenumbers or frequencies.
    gamma : float or array_like
        Damping constant. If an array, size must match omega_p and eps_inf.
    omega_p : float or array_like
        Plasma frequency. If an array, size must match gamma and eps_inf.
    eps_inf : float or array_like
        High-frequency permittivity limit. If an array, size must match omega_p and gamma.

    Returns
    -------
    complex or ndarray
        The calculated complex relative permittivity. Axes of size one are removed.
    """
    if not np.iterable(omega):
        omega = np.asarray([omega])
    if len(omega.shape) != 1:
        raise ValueError("omega must be a scalar or a 1-D array of wavenumbers")
    num_wavenumbers = len(omega)

    if not np.iterable(gamma):
        gamma = np.asarray([gamma])
    if len(gamma.shape) != 1:
        raise ValueError("gamma must be a scalar or a 1-D array of damping coefficients")

    if not np.iterable(omega_p):
        omega_p = np.asarray([omega_p])
    if len(omega_p.shape) != 1:
        raise ValueError("omega_p must be a scalar or a 1-D array of plasma frequencies")

    if not np.iterable(eps_inf):
        eps_inf = np.asarray([eps_inf])
    if len(eps_inf.shape) != 1:
        raise ValueError("eps_inf must be a scalar or a 1-D array of high frequency permitivities")

    if np.any(len(eps_inf) != np.array([len(gamma), len(omega_p)])):
        raise ValueError("eps_inf, gamma, and omega_p must all have the same number of values applied")
    num_particles = len(eps_inf)

    eps_p = np.zeros([num_wavenumbers, num_particles], dtype=np.complex128)
    eps_p[...] = eps_inf[None, :] - omega_p[None, :]**2 / (omega[:, None]**2 + 1j * omega[:, None] * gamma[None, :])
    return np.squeeze(eps_p)

class dipole_solver:
    """
    Python wrapper for the CUDA-accelerated C++ Dipole_Solver class.

    This class computes the self-consistent induced dipoles of a system of particles 
    subject to an external electric field under periodic boundary conditions. It leverages 
    a GPU-resident restarted complex GMRES or BiCGSTAB solver and uses 3D Ewald summation 
    to evaluate the dipole-dipole interactions efficiently.
    """
    def __init__(self, box, eps_p, radius=1.0, eps_m=1.0, xi=0.5, tol=1e-3, quiet=False, guess_type="derivative", solver_type="gmres", field_type="auto", E0=None, quadrupoles=False):
        """
        Initialize the dipole solver with system and solver parameters.

        Parameters
        ----------
        box : array_like
            The dimensions of the periodic boundary box [L_x, L_y, L_z] in nanometers.
        eps_p : complex, array_like or nested lists
            The complex relative permittivity (dielectric constant) of the particles.
        radius : float or array_like, optional
            The radii of the particles in nanometers. Defaults to 1.0.
        eps_m : float, optional
            The relative permittivity of the surrounding medium. Defaults to 1.0.
        xi : float, optional
            The Ewald splitting parameter. Defaults to 0.5.
        tol : float, optional
            The numerical solver tolerance for relative error convergence. Defaults to 1e-3.
        quiet : bool, optional
            If True, suppresses solver debug output. Defaults to False.
        guess_type : str, optional
            Initial guess type: "zero", "static", or "derivative". Defaults to "derivative".
        solver_type : str, optional
            Krylov solver type: "gmres" or "bicgstab". Defaults to "gmres".
        field_type : str, optional
            Field grid type: "auto", "monodisperse", "polydisperse", or "direct" (open BCs). Defaults to "auto".
        E0 : array_like, optional
            The electric polarization of the incident field (a 3-element vector or a list/array of 3-element vectors). Defaults to None.
        """
        if field_type not in ("auto", "monodisperse", "polydisperse", "direct"):
            raise ValueError(f"field_type must be 'auto', 'monodisperse', 'polydisperse', or 'direct', got {field_type}")

        # Convert box to a list of 3 floats
        box_list = [float(x) for x in box]
        if len(box_list) != 3:
            raise ValueError("box must have length 3")

        # Convert radius to a list of floats
        if np.iterable(radius):
            radius_list = [float(r) for r in radius]
        else:
            radius_list = [float(radius)]

        # Convert E0 to standard 2D list of complex
        E0_list = []
        if E0 is not None:
            E0_arr = np.asarray(E0)
            if E0_arr.ndim == 1:
                if E0_arr.size % 3 != 0:
                    raise ValueError("Incident field vector size must be a multiple of 3")
                E0_list = [[complex(x) for x in E0_arr]]
            elif E0_arr.ndim == 2:
                if E0_arr.shape[1] == 3:
                    E0_list = [[complex(x) for x in row] for row in E0_arr]
                elif E0_arr.shape[1] % 3 == 0:
                    E0_list = [[complex(x) for x in row] for row in E0_arr]
                else:
                    raise ValueError("Each vector in E0 must be 3-dimensional or have a size that is a multiple of 3")
            elif E0_arr.ndim == 3:
                if E0_arr.shape[2] != 3:
                    raise ValueError("The last dimension of E0 must be 3")
                E0_list = [[complex(x) for x in block.ravel()] for block in E0_arr]
            else:
                raise ValueError("E0 must be a 1D, 2D, or 3D array")

        # Resolve quadrupoles argument
        solve_quadrupoles = False
        quad_idxs_list = []
        if isinstance(quadrupoles, bool):
            solve_quadrupoles = quadrupoles
        elif np.iterable(quadrupoles):
            solve_quadrupoles = True
            quad_idxs_list = [int(i) for i in quadrupoles]
        else:
            raise TypeError("quadrupoles must be a boolean or a list/iterable of particle indices")

        # Convert eps_p based on dimension
        eps_p_arr = np.asarray(eps_p)
        if eps_p_arr.ndim == 0:
            # Scalar
            eps_p_scalar = complex(eps_p_arr.item())
            self._solver = _cuMPM.Dipole_Solver(
                box_list, eps_p_scalar, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, E0_list, solve_quadrupoles, quad_idxs_list
            )
        elif eps_p_arr.ndim == 1:
            # 1D array
            if eps_p_arr.size == 1:
                eps_p_scalar = complex(eps_p_arr.item())
                self._solver = _cuMPM.Dipole_Solver(
                    box_list, eps_p_scalar, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, E0_list, solve_quadrupoles, quad_idxs_list
                )
            else:
                eps_p_1d = [complex(x) for x in eps_p_arr]
                self._solver = _cuMPM.Dipole_Solver(
                    box_list, eps_p_1d, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, E0_list, solve_quadrupoles, quad_idxs_list
                )
        elif eps_p_arr.ndim == 2:
            # 2D array
            eps_p_2d = [[complex(x) for x in row] for row in eps_p_arr]
            self._solver = _cuMPM.Dipole_Solver(
                box_list, eps_p_2d, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, E0_list, solve_quadrupoles, quad_idxs_list
            )
        else:
            raise ValueError("eps_p must be a scalar, 1D array, or 2D array")

    def compute(self, positions):
        """
        Calculate the dipoles for the positions given.
        
        Parameters
        ----------
        positions : array_like
            An array of shape (num_particles, 3) or (num_frames, num_particles, 3)
        """
        positions = np.asarray(positions)
        if positions.ndim == 2:
            positions = positions[np.newaxis, ...]
        elif positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(
                "positions must be of shape (num_particles, 3) or (num_frames, num_particles, 3)"
            )

        num_frames = positions.shape[0]
        for frame_idx in range(num_frames):
            x_part = positions[frame_idx, :, 0].tolist()
            y_part = positions[frame_idx, :, 1].tolist()
            z_part = positions[frame_idx, :, 2].tolist()
            self._solver.compute(x_part, y_part, z_part)

    def get_eff_polarizability(self, physical=True):
        """
        Returns the effective polarizability by averaging all particle dipoles over all frames
        
        Parameters
        ----------
        physical : bool, optional
            If True, returns values scaled back to physical units. If False, returns dimensionless values. Defaults to True.

        Returns
        -------
        alpha_eff : numpy.ndarray
            Average polarizability with shape (num_wavelengths, 3, 3) (squeezed)
        """
        return np.squeeze(self._solver.get_eff_polarizability(physical))

    def get_dipoles(self, physical=True):
        """
        Returns the dipoles calculated for all frames.
        
        Parameters
        ----------
        physical : bool, optional
            If True, returns values scaled back to physical units. If False, returns dimensionless values. Defaults to True.

        Returns
        -------
        p : numpy.ndarray
            Dipoles of each particle with shape (num_frames, num_wavelengths, num_particles, 3, 3) (squeezed)
        """
        return np.squeeze(self._solver.get_dipoles(physical))

    def get_quadrupoles(self, physical=True):
        """
        Returns the quadrupoles calculated for all frames.
        
        Parameters
        ----------
        physical : bool, optional
            If True, returns values scaled back to physical units. If False, returns dimensionless values. Defaults to True.

        Returns
        -------
        Q : numpy.ndarray
            Quadrupoles of each particle in the quadrupole subset with shape (num_frames, num_wavelengths, num_quads, 3, 5) (squeezed)
        """
        return np.squeeze(self._solver.get_quadrupoles(physical))

    def get_cap_dip(self):
        """
        Deprecated. Use get_dipoles and get_eff_polarizability instead.
        """
        return self.get_eff_polarizability(), self.get_dipoles()
