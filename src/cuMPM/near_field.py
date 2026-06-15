import numpy as np
from . import _cuMPM

class Near_Field:
    r"""
    Evaluate the local electric field intensity at arbitrary spatial coordinates.

    This class computes the total local electric field intensity (defined as 
    :math:`\lvert -\mathbf{E}_{\text{ind}} + \mathbf{E}_0 \rvert^2`) at a set of target field points 
    due to a collection of complex point dipoles under periodic boundary conditions. 
    It supports both monodisperse and polydisperse systems.
    """
    def __init__(self, box, E0, radius=1.0, dip=None, dip_pos=None, field_points=None, xi=0.5, errortol=1e-3, field_type="auto"):
        """
        Initialize the Near Field calculator.

        Parameters
        ----------
        box : array_like
            The dimensions of the periodic boundary box [L_x, L_y, L_z] in nanometers.
        E0 : array_like
            The complex incident electric field vector [E_x, E_y, E_z].
        radius : float or array_like, optional
            The radii of the particles in nanometers. Defaults to 1.0.
        dip : array_like, optional
            The complex dipoles of each particle, shape (num_particles, 3).
        dip_pos : array_like, optional
            The spatial positions of the dipoles, shape (num_particles, 3).
        field_points : array_like, optional
            The target spatial coordinates for field evaluation, shape (num_field_points, 3).
        xi : float, optional
            The Ewald splitting parameter. Defaults to 0.5.
        errortol : float, optional
            The real space cutoff error tolerance. Defaults to 1e-3.
        field_type : str, optional
            Field grid type: "auto", "monodisperse", or "polydisperse". Defaults to "auto".
        """
        self.box = [float(x) for x in box]
        self.E0 = [complex(x) for x in E0]
        
        # Convert radius to list
        if np.iterable(radius):
            self.radius_list = [float(r) for r in radius]
        else:
            self.radius_list = [float(radius)]

        self.xi = float(xi)
        self.errortol = float(errortol)
        self.field_type = field_type

        # Instantiate C++ class
        self._near_field = _cuMPM.Near_Field(
            self.box, self.E0, self.radius_list, self.xi, self.errortol, self.field_type
        )

        if dip is not None:
            self.set_dipoles(dip)
        if dip_pos is not None:
            self.set_dipole_positions(dip_pos)
        if field_points is not None:
            self.set_field_points(field_points)

    def set_dipoles(self, dip):
        """
        Set the complex dipoles of the particles.

        Parameters
        ----------
        dip : array_like
            Array of complex dipole vectors, shape (num_particles, 3).
        """
        if dip is None:
            raise ValueError("dipoles can't be set to None")
        dip_arr = np.asarray(dip, dtype=complex)
        if dip_arr.ndim == 2:
            dip_flat = dip_arr.ravel().tolist()
        else:
            dip_flat = [complex(x) for x in dip_arr]
        self._near_field.set_dipoles(dip_flat)

    def set_dipole_positions(self, dip_pos):
        """
        Set the spatial coordinates of the dipoles.

        Parameters
        ----------
        dip_pos : array_like
            Array of spatial coordinates [x, y, z] for each dipole, shape (num_particles, 3).
        """
        if dip_pos is None:
            raise ValueError("dipole positions can't be set to None")
        dip_pos_arr = np.asarray(dip_pos, dtype=float)
        x = dip_pos_arr[:, 0].tolist()
        y = dip_pos_arr[:, 1].tolist()
        z = dip_pos_arr[:, 2].tolist()
        self._near_field.set_dipole_positions(x, y, z)

    def set_field_points(self, field_points):
        """
        Set the target spatial coordinates at which to calculate the electric field.

        Parameters
        ----------
        field_points : array_like
            Array of coordinates [x, y, z] where the field is evaluated, shape (num_field_points, 3).
        """
        if field_points is None:
            raise ValueError("field points can't be set to None")
        field_points_arr = np.asarray(field_points, dtype=float)
        x = field_points_arr[:, 0].tolist()
        y = field_points_arr[:, 1].tolist()
        z = field_points_arr[:, 2].tolist()
        self._near_field.set_field_points(x, y, z)

    def calculate(self):
        """
        Evaluate the local electric field intensity at the pre-configured field points.

        Returns
        -------
        intensity : numpy.ndarray
            A 1D array of shape (num_field_points,) representing the total field 
            intensity ``|-E_ind + E0|^2`` at each evaluation coordinate.
        """
        intensity = self._near_field.calculate()
        return np.array(intensity)
