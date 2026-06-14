import numpy as np
from . import _cuMPM

class dipole_solver:
    """
    Python wrapper for the CUDA-accelerated C++ Dipole_Solver class.
    Accepts Python lists or NumPy arrays.
    """
    def __init__(self, box, eps_p, radius=1.0, eps_m=1.0, xi=0.5, tol=1e-3, quiet=False, guess_type="derivative"):
        # Convert box to a list of 3 floats
        box_list = [float(x) for x in box]
        if len(box_list) != 3:
            raise ValueError("box must have length 3")

        # Convert radius to a list of floats
        if np.iterable(radius):
            radius_list = [float(r) for r in radius]
        else:
            radius_list = [float(radius)]

        # Convert eps_p based on dimension
        eps_p_arr = np.asarray(eps_p)
        if eps_p_arr.ndim == 0:
            # Scalar
            eps_p_scalar = complex(eps_p_arr.item())
            self._solver = _cuMPM.Dipole_Solver(
                box_list, eps_p_scalar, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type
            )
        elif eps_p_arr.ndim == 1:
            # 1D array
            if eps_p_arr.size == 1:
                eps_p_scalar = complex(eps_p_arr.item())
                self._solver = _cuMPM.Dipole_Solver(
                    box_list, eps_p_scalar, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type
                )
            else:
                eps_p_1d = [complex(x) for x in eps_p_arr]
                self._solver = _cuMPM.Dipole_Solver(
                    box_list, eps_p_1d, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type
                )
        elif eps_p_arr.ndim == 2:
            # 2D array
            eps_p_2d = [[complex(x) for x in row] for row in eps_p_arr]
            self._solver = _cuMPM.Dipole_Solver(
                box_list, eps_p_2d, radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type
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
