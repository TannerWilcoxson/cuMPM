import numpy as np
from .electric_field import Electric_Field
from .neighbor_list import MultiNeighborList

class Near_Field:
    '''
    Calculate the electric field at *points* for the given *dipoles* at *dip_pos*.

    **Parameters**
        *box*
            A 1D array of length three specifying *Lx*, *Ly*, and *Lz*.
        *E0*
            The incident field. 1D array of length three.
        *radius*
            The radius of the dipole particles. Defaults to 1 (assumed nondimentionalized).
        *dip*
            A 2D array (num_dipoles,3) specifying the dipoles contributing to the electric field in units of eps_m*|E_0|*R**3 (the same as the MPM class returns).
            Can also be set with **set_dipoles** method, but must be set before calling **calculate**.
        *dip_pos*
            A 2D array (num_dipoles,3) specifying the position of the dipoles contributing to the electric field.
            Can also be set with **set_dip_pos** method, but must be set before calling **calculate**.
        *field_points*
            A 2D (num_field_points, 3) array specifying the field_points points where the electric field_points should be calculated.
            Can also be set with **set_field_points** method, but must be set before calling **calculate**.
        *xi*
            Ewald parameter. Defaults to 0.5
        *errortol*
            The error tolerance. Defaults to 0.001.
    '''
    def __init__(self,box,E0,radius=1,dip=None,dip_pos=None,field_points=None,xi=0.5,errortol=1e-3):# {{{
        self.dip = dip
        self.dip_pos = dip_pos
        self.E0 = E0
        self.xi = xi
        self.errortol = errortol

        self.radius = radius

        self.box = box
        self.field_calculator = None

        self.field_update = True
        self.dip_pos_update = True
        self.dip_update = True
        return# }}}
    def set_dipoles(self,dip):# {{{
        '''
        Used to set or change the dipoles.

        **Parameters**
            *dip*
                Dipoles contributing to the electric field.
        '''
        if dip is None:
            raise Exception("dipoles can't be set to None")
        self.dip = dip
        self.dip_update = True# }}}
    def set_dipole_positions(self,dip_pos):# {{{
        '''
        Used to set or change the dipole positions.

        **Parameters**
            *dip_pos*
                The position of the dipoles contributing to the electric field.
        '''
        if dip_pos is None:
            raise Exception("dipole positions can't be set to None")
        self.dip_pos = dip_pos
        self.dip_pos_update = True# }}}
    def set_field_points(self,field_points):# {{{
        '''
        Used to set or change the field points.

        **Parameters**
            *field_points*
                The field points where the electric field should be calculated.
        '''
        if field_points is None:
            raise Exception("field points can't be set to None")
        self.field_points = field_points
        self.field_update = True
        # }}}
    def calculate(self,):# {{{
        '''
        Calculates the electric field intensity at each field points.

        **Returns**
            *E*
                The electric field intensity at each field point.
        '''
        self._check_have_data()
        if self.field_calculator is None:
            number_particles = self.dip.shape[0]
            self._set_dims(number_particles)
            self._nondimensionalize()
            self.field_calculator = Electric_Field(self.box,self.xi,self.errortol)
        self._update()
        E = -self.field_calculator.calculate() + self.E0[None,:]
        E = np.sum(np.abs(E)**2,axis = 1)
        return E
        # }}}
    def _set_dims(self,num_p):# {{{
        self.num_particles = num_p

        if not np.iterable(self.radius):
            self.radius = self.radius*np.ones(num_p)
        self.radius = np.asarray(self.radius)
        self.radius = np.copy(self.radius)
        if len(self.radius.shape) != 1:
            raise Exception("radius must be passed as a scalar or 1-D array of length num_particles")
        elif len(self.radius) != num_p:
            raise Exception("The number of dipoles provided by dip_pos and dipoles is inconsistent with the number of radii provided")
        elif not np.all(self.radius == self.radius[0]):
            return NotImplementedError("Radii of different sizes not yet supported")
        # }}}
    def _update(self):# {{{
        if self.field_update:
            self.field_points = self.field_points/self.length_scale
            self.field_calculator.set_points(self.field_points)

        if self.dip_pos_update:
            self.dip_pos = self.dip_pos/self.length_scale
            self.field_calculator.set_dip_pos(self.dip_pos)

        if self.dip_update:
            self.field_calculator.set_dipoles(self.dip)

        if self.dip_update and self.dip_pos_update:
            if self.dip.shape[0] != self.dip_pos.shape[0]:
                raise Exception("The first dimension of dipole and dipole position arrays must match")

        self.field_update = False
        self.dip_pos_update = False
        self.dip_update = False
        return# }}}
    def _check_have_data(self,):# {{{
        if self.field_points is None:
            raise Exception("field_points must be set before calculation")
        if self.dip_pos is None:
            raise Exception("dipole positions must be set before calculation")
        if self.dip is None:
            raise Exception("dipoles must be set before calculation")# }}}
    def _nondimensionalize(self,):# {{{
        self.length_scale = self.radius[0]
        self.box = self.box/self.length_scale
    # }}}
