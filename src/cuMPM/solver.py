import numpy as np
from . import _cuMPM

class dipole_solver:
    """
    Python wrapper for the CUDA-accelerated C++ Dipole_Solver class.

    This class computes the self-consistent induced dipoles of a system of particles 
    subject to an external electric field under periodic boundary conditions. It leverages 
    a GPU-resident restarted complex GMRES or BiCGSTAB solver and uses 3D Ewald summation 
    to evaluate the dipole-dipole interactions efficiently.
    """
    def __init__(self, box, eps_p, radius=1.0, eps_m=1.0, xi=0.5, tol=1e-3, quiet=False, guess_type="derivative", solver_type="gmres", field_type="auto", E0=None):
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
                if E0_arr.size != 3:
                    raise ValueError("Incident field vector must be 3-dimensional")
                E0_list = [[complex(x) for x in E0_arr]]
            elif E0_arr.ndim == 2:
                if E0_arr.shape[1] != 3:
                    raise ValueError("Each incident field vector must be 3-dimensional")
                E0_list = [[complex(x) for x in row] for row in E0_arr]
            else:
                raise ValueError("E0 must be a 1D vector or a 2D array of vectors")

        # Convert eps_p based on dimension
        eps_p_arr = np.asarray(eps_p)
        if eps_p_arr.ndim == 0:
            # Scalar
            eps_p_scalar = complex(eps_p_arr.item())
            self._solver = _cuMPM.Dipole_Solver(
                box_list, eps_p_scalar, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, E0_list
            )
        elif eps_p_arr.ndim == 1:
            # 1D array
            if eps_p_arr.size == 1:
                eps_p_scalar = complex(eps_p_arr.item())
                self._solver = _cuMPM.Dipole_Solver(
                    box_list, eps_p_scalar, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, E0_list
                )
            else:
                eps_p_1d = [complex(x) for x in eps_p_arr]
                self._solver = _cuMPM.Dipole_Solver(
                    box_list, eps_p_1d, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, E0_list
                )
        elif eps_p_arr.ndim == 2:
            # 2D array
            eps_p_2d = [[complex(x) for x in row] for row in eps_p_arr]
            self._solver = _cuMPM.Dipole_Solver(
                box_list, eps_p_2d, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, E0_list
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

    def get_eff_polarizability(self):
        """
        Returns the effective polarizability by averaging all particle dipoles over all frames.
        
        Returns
        -------
        alpha_eff : numpy.ndarray
            Average polarizability with shape (num_wavelengths, 3, 3) (squeezed)
        """
        return np.squeeze(self._solver.get_eff_polarizability())

    def get_dipoles(self):
        """
        Returns the dipoles calculated for all frames.
        
        Returns
        -------
        p : numpy.ndarray
            Dipoles of each particle with shape (num_frames, num_wavelengths, num_particles, 3, 3) (squeezed)
        """
        return np.squeeze(self._solver.get_dipoles())

    def get_cap_dip(self):
        """
        Deprecated. Use get_dipoles and get_eff_polarizability instead.
        """
        return self.get_eff_polarizability(), self.get_dipoles()
