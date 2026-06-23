import numpy as np
from . import _cuMPM

class EELS:
    """
    Python wrapper for the CUDA-accelerated C++ EELS_Solver class.

    This class computes the Electron Energy Loss Spectroscopy (EELS) probability 
    spectrum and self-consistent induced dipoles for a collection of particles 
    subject to the relativistic incident field of a moving electron beam.
    """
    def __init__(self, box, eps_p, omega, v, eps_m=1.0, radius=1.0, xi=0.5, tol=1e-3, quiet=False, guess_type="derivative", solver_type="gmres", field_type="auto"):
        """
        Initialize the EELS solver with system, electron, and solver parameters.

        Parameters
        ----------
        box : array_like
            The dimensions of the boundary box [L_x, L_y, L_z] in nanometers.
        eps_p : complex, array_like or nested lists
            The complex relative permittivity (dielectric constant) of the particles.
        omega : array_like
            The wavenumbers at which to calculate EELS in 1/nm.
        v : float
            The velocity of the electron beam as a fraction of the speed of light c (0 < v < 1).
        eps_m : float, optional
            The relative permittivity of the surrounding medium. Defaults to 1.0.
        radius : float or array_like, optional
            The radii of the particles in nanometers. Defaults to 1.0.
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
            Field evaluation type: "auto", "monodisperse", "polydisperse", or "direct" (open BCs). Defaults to "auto".
        """
        if field_type not in ("auto", "monodisperse", "polydisperse", "direct"):
            raise ValueError(f"field_type must be 'auto', 'monodisperse', 'polydisperse', or 'direct', got {field_type}")

        box_list = [float(x) for x in box]
        if len(box_list) != 3:
            raise ValueError("box must have length 3")

        omega_list = [float(o) for o in omega]

        if np.iterable(radius):
            radius_list = [float(r) for r in radius]
        else:
            radius_list = [float(radius)]

        # Convert eps_p based on dimension
        eps_p_arr = np.asarray(eps_p)
        if eps_p_arr.ndim == 0:
            eps_p_scalar = complex(eps_p_arr.item())
            self._solver = _cuMPM.EELS_Solver(
                box_list, eps_p_scalar, omega_list, float(v), radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type
            )
        elif eps_p_arr.ndim == 1:
            if eps_p_arr.size == 1:
                eps_p_scalar = complex(eps_p_arr.item())
                self._solver = _cuMPM.EELS_Solver(
                    box_list, eps_p_scalar, omega_list, float(v), radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type
                )
            else:
                eps_p_1d = [complex(x) for x in eps_p_arr]
                self._solver = _cuMPM.EELS_Solver(
                    box_list, eps_p_1d, omega_list, float(v), radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type
                )
        elif eps_p_arr.ndim == 2:
            eps_p_2d = [[complex(x) for x in row] for row in eps_p_arr]
            self._solver = _cuMPM.EELS_Solver(
                box_list, eps_p_2d, omega_list, float(v), radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type
            )
        else:
            raise ValueError("eps_p must be a scalar, 1D array, or 2D array")

    def compute(self, epos, positions):
        """
        Calculate the EELS probability and induced dipoles for the given configurations.

        Parameters
        ----------
        epos : array_like
            The 2D electron beam impact coordinates, shape (2,) or (num_frames, 2).
        positions : array_like
            The spatial coordinates of the particles, shape (num_particles, 3) or (num_frames, num_particles, 3).
        """
        positions = np.asarray(positions)
        if positions.ndim == 2:
            positions = positions[np.newaxis, ...]
        elif positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must be of shape (num_particles, 3) or (num_frames, num_particles, 3)")

        epos = np.asarray(epos)
        if epos.ndim == 1:
            epos = epos[np.newaxis, ...]
        elif epos.ndim != 2 or epos.shape[-1] != 2:
            raise ValueError("epos must be of shape (2,) or (num_frames, 2)")

        num_frames = positions.shape[0]
        if epos.shape[0] != num_frames:
            raise ValueError(f"The number of frames in epos ({epos.shape[0]}) must match positions ({num_frames})")

        for frame_idx in range(num_frames):
            x_part = positions[frame_idx, :, 0].tolist()
            y_part = positions[frame_idx, :, 1].tolist()
            z_part = positions[frame_idx, :, 2].tolist()
            epos_frame = epos[frame_idx].tolist()
            self._solver.compute(epos_frame, x_part, y_part, z_part)

    def get_eels(self):
        """
        Returns the computed EELS probability spectrum.

        Returns
        -------
        eels : numpy.ndarray
            EELS probability spectrum with shape (num_frames, num_wavelengths) (squeezed if single frame)
        """
        return np.squeeze(self._solver.get_eels())

    def get_dipoles(self):
        """
        Returns the computed induced dipoles.

        Returns
        -------
        p : numpy.ndarray
            Induced dipoles of each particle with shape (num_frames, num_wavelengths, num_particles, 3) (squeezed)
        """
        return np.squeeze(self._solver.get_dipoles())
