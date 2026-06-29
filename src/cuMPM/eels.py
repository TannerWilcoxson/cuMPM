import numpy as np
from . import _cuMPM

class EELS:
    """
    Python wrapper for the CUDA-accelerated C++ EELS_Solver class.

    This class computes the Electron Energy Loss Spectroscopy (EELS) probability 
    spectrum and self-consistent induced dipoles for a collection of particles 
    subject to the relativistic incident field of a moving electron beam.
    """
    def __init__(self, box, eps_p, omega, v, eps_m=1.0, radius=1.0, xi=0.5, tol=1e-3, quiet=False, guess_type="derivative", solver_type="gmres", field_type="auto", split_dist=None, N_split=None, integration_step=0.05, split_method="cubic", solve_quadrupoles=False, quad_idxs=None):
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
        split_dist : float, optional
            The cutoff distance from the electron beam within which to split particles. Defaults to None.
        N_split : int, optional
            The target number of sub-dipoles to split each close particle into. Defaults to None.
        integration_step : float, optional
            Step size along the electron beam trajectory in nanometers. Defaults to 0.05.
        split_method : str, optional
            Algorithm used to distribute sub-dipoles inside the NC volume.
            ``"cubic"``    – uniform cubic lattice truncated at the NC radius (default).
                            N_split is rounded up to the next perfect cube n³; the actual
                            sub-dipole count M ≈ (π/6)·n³ after truncation.
            ``"fibonacci"``– Fibonacci/golden-angle spiral on the unit sphere with
                            cube-root-spaced shells decoupled by a fixed random shuffle.
                            The actual count equals exactly N_split.
            In both cases sub-dipole radii are set so that the total sub-dipole volume
            equals the original NC volume.
        solve_quadrupoles : bool, optional
            Whether to solve for quadrupoles as well. Defaults to False.
        quad_idxs : list of int, optional
            A list of particle indices for which quadrupoles should be solved. If None or empty, all particles are solved.
        """
        if split_method not in ("cubic", "fibonacci"):
            raise ValueError("split_method must be 'cubic' or 'fibonacci', got " + repr(split_method))
        self.split_method = split_method

        if (split_dist is None) != (N_split is None):
            raise ValueError("Both split_dist and N_split must be specified, or both left as None.")

        if split_dist is None:
            self.split_dist = 0.0
            self.N_split = 0
        else:
            self.split_dist = float(split_dist)
            self.N_split = int(N_split)

        # Store parameters for dynamic solver instantiation in compute() if splitting is enabled
        self.box = box
        self.eps_p = eps_p
        self.omega = omega
        self.v = v
        self.eps_m = eps_m
        self.radius = radius
        self.xi = xi
        self.tol = tol
        self.quiet = quiet
        self.guess_type = guess_type
        self.solver_type = solver_type
        self.field_type = field_type
        self.integration_step = float(integration_step)
        self.solve_quadrupoles = bool(solve_quadrupoles)
        self.quad_idxs = [int(i) for i in quad_idxs] if quad_idxs is not None else []

        # Initialize results lists for splitting mode
        self._eels_results = []
        self._dips_results = []
        self._quads_results = []
        self._split_poss = []
        self.positions = None

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

        if self.split_dist > 0.0 and self.N_split > 1:
            self._solver = None
        else:
            # Convert eps_p based on dimension
            eps_p_arr = np.asarray(eps_p)
            if eps_p_arr.ndim == 0:
                eps_p_scalar = complex(eps_p_arr.item())
                self._solver = _cuMPM.EELS_Solver(
                    box_list, eps_p_scalar, omega_list, float(v), radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, float(integration_step), self.solve_quadrupoles, self.quad_idxs
                )
            elif eps_p_arr.ndim == 1:
                if eps_p_arr.size == 1:
                    eps_p_scalar = complex(eps_p_arr.item())
                    self._solver = _cuMPM.EELS_Solver(
                        box_list, eps_p_scalar, omega_list, float(v), radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, float(integration_step), self.solve_quadrupoles, self.quad_idxs
                    )
                else:
                    eps_p_1d = [complex(x) for x in eps_p_arr]
                    self._solver = _cuMPM.EELS_Solver(
                        box_list, eps_p_1d, omega_list, float(v), radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, float(integration_step), self.solve_quadrupoles, self.quad_idxs
                    )
            elif eps_p_arr.ndim == 2:
                eps_p_2d = [[complex(x) for x in row] for row in eps_p_arr]
                self._solver = _cuMPM.EELS_Solver(
                    box_list, eps_p_2d, omega_list, float(v), radius_list, float(eps_m), float(xi), float(tol), bool(quiet), guess_type, solver_type, field_type, float(integration_step), self.solve_quadrupoles, self.quad_idxs
                )
            else:
                raise ValueError("eps_p must be a scalar, 1D array, or 2D array")

    # ------------------------------------------------------------------
    # Sub-dipole lattice generators
    # ------------------------------------------------------------------

    def _make_cubic_offsets(self):
        """
        Return unit-sphere offsets on a cubic lattice truncated at r=1.

        N_split is rounded up to the next perfect cube n³ to determine grid
        resolution n.  Points in [-1,1]³ whose distance from the origin
        exceeds 1 are discarded.  The retained count M ≈ (π/6)·n³.
        """
        n = int(np.ceil(self.N_split ** (1.0 / 3.0)))
        while n ** 3 < self.N_split:
            n += 1

        h = 2.0 / n
        coords_1d = np.linspace(-1.0 + h / 2.0, 1.0 - h / 2.0, n)
        gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        inside = np.linalg.norm(grid, axis=1) <= 1.0
        return grid[inside]

    def _make_fibonacci_offsets(self):
        """
        Return N_split unit-sphere offsets using a Fibonacci/golden-angle spiral.

        Directions are drawn from the Fibonacci sphere algorithm (uniform surface
        coverage).  Shell radii use cube-root spacing for uniform volume density.
        The two are decoupled by a fixed random shuffle so that no particular shell
        radius is tied to a particular latitude.
        """
        N = self.N_split
        phi = np.pi * (3.0 - np.sqrt(5.0))

        directions = np.empty((N, 3))
        for i in range(N):
            y_unit  = 1.0 - (i / float(N - 1)) * 2.0
            theta   = phi * i
            sin_lat = np.sqrt(max(0.0, 1.0 - y_unit * y_unit))
            directions[i] = [np.cos(theta) * sin_lat, y_unit, np.sin(theta) * sin_lat]

        r_fracs = np.array([((i + 0.5) / N) ** (1.0 / 3.0) for i in range(N)])
        rng = np.random.default_rng(0)
        rng.shuffle(r_fracs)

        return r_fracs[:, np.newaxis] * directions

    def _split_close_dipoles(self, positions, e_pos, radii, eps_p_2d):
        """
        Split particles close to the electron beam into sub-dipoles.

        The distribution method is controlled by ``self.split_method``:
        - ``"cubic"``     – cubic lattice truncated at the NC sphere boundary.
        - ``"fibonacci"`` – Fibonacci volume lattice (golden-angle directions +
                            cube-root shells, radially decoupled).

        In both cases sub-dipole radii are chosen to conserve the total NC volume:
            M × (4π/3) sub_R³ = (4π/3) R³  ⟹  sub_R = R / M^(1/3)
        """
        dists_2d = np.linalg.norm(positions[:, :2] - e_pos[np.newaxis, :], axis=1)
        to_split = dists_2d < self.split_dist

        if not np.any(to_split) or self.N_split <= 1:
            return positions.copy(), radii.copy(), eps_p_2d.copy()

        new_positions = []
        new_radii = []
        new_eps_p = []

        # Keep unsplit particles
        unsplit_indices = np.where(~to_split)[0]
        for idx in unsplit_indices:
            new_positions.append(positions[idx])
            new_radii.append(radii[idx])
            new_eps_p.append(eps_p_2d[:, idx])

        # Build the unit-sphere offsets using the chosen method
        if self.split_method == "cubic":
            unit_offsets = self._make_cubic_offsets()
        else:
            unit_offsets = self._make_fibonacci_offsets()

        # Number of sub-dipoles (exact for Fibonacci; ≈π/6·n³ for cubic)
        M = len(unit_offsets)

        # Sub-dipole radius conserves total NC volume:
        #   M × (4π/3) sub_R³ = (4π/3) R³  →  sub_R = R / M^(1/3)
        sub_R_frac = 1.0 / (M ** (1.0 / 3.0))

        # Split close particles
        split_indices = np.where(to_split)[0]
        for idx in split_indices:
            R   = radii[idx]
            pos = positions[idx]
            eps = eps_p_2d[:, idx]

            sub_pos = pos + unit_offsets * R
            sub_R   = R * sub_R_frac

            for sp in sub_pos:
                new_positions.append(sp)
                new_radii.append(sub_R)
                new_eps_p.append(eps)

        final_positions = np.array(new_positions)
        final_radii     = np.array(new_radii)
        final_eps_p     = np.column_stack(new_eps_p)

        return final_positions, final_radii, final_eps_p

    def compute(self, epos, positions):
        """
        Calculate the EELS probability and induced dipoles for the given configurations.

        Parameters
        ----------
        epos : array_like
            The 2D electron beam impact coordinates, shape (2,) or (num_epos, 2).
        positions : array_like
            The spatial coordinates of the particles, shape (num_particles, 3).
        """
        positions = np.asarray(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be of shape (num_particles, 3)")

        epos = np.asarray(epos)
        if epos.ndim == 1:
            if epos.shape[0] != 2:
                raise ValueError("epos must be of shape (2,) or (num_epos, 2)")
            epos = epos[np.newaxis, :]
        elif epos.ndim != 2 or epos.shape[1] != 2:
            raise ValueError("epos must be of shape (2,) or (num_epos, 2)")

        num_epos = epos.shape[0]
        num_particles = positions.shape[0]
        self.positions = positions

        if self.split_dist > 0.0 and self.N_split > 1:
            # Clear previous results from this compute call
            self._eels_results = []
            self._dips_results = []
            self._split_poss = []

            # We must standardize radii and eps_p to map to individual particles
            if np.iterable(self.radius):
                radii = np.asarray(self.radius, dtype=np.float64)
                if radii.ndim == 0:
                    radii = np.repeat(radii.item(), num_particles)
                elif radii.size == 1:
                    radii = np.repeat(radii.item(), num_particles)
                elif radii.size != num_particles:
                    raise ValueError("The number of particles is inconsistent with the number of radii provided.")
            else:
                radii = np.repeat(float(self.radius), num_particles)

            eps_p_arr = np.asarray(self.eps_p)
            num_wavevectors = len(self.omega)
            if eps_p_arr.ndim == 0:
                eps_p_2d = np.full((num_wavevectors, num_particles), complex(eps_p_arr.item()))
            elif eps_p_arr.ndim == 1:
                if eps_p_arr.size == 1:
                    eps_p_2d = np.full((num_wavevectors, num_particles), complex(eps_p_arr.item()))
                else:
                    if eps_p_arr.size != num_wavevectors:
                        raise ValueError("1D eps_p must have length equal to num_wavevectors")
                    eps_p_2d = np.tile(eps_p_arr[:, np.newaxis], (1, num_particles))
            elif eps_p_arr.ndim == 2:
                if eps_p_arr.shape[0] != num_wavevectors:
                    raise ValueError("First dimension of 2D eps_p must be equal to num_wavevectors")
                if eps_p_arr.shape[1] == 1:
                    eps_p_2d = np.tile(eps_p_arr, (1, num_particles))
                elif eps_p_arr.shape[1] != num_particles:
                    raise ValueError("Second dimension of 2D eps_p must be equal to num_particles")
                else:
                    eps_p_2d = eps_p_arr.copy()
            else:
                raise ValueError("eps_p must be a scalar, 1D array, or 2D array")

            # Loop over all electron beam impact coordinates
            for epos_idx in range(num_epos):
                e_pos = epos[epos_idx]
                if not self.quiet:
                    print(f"  [EELS] Processing impact parameter {epos_idx + 1}/{num_epos}...")
                
                # Split particles that are within split_dist from the ebeam
                split_pos, split_R, split_eps = self._split_close_dipoles(positions, e_pos, radii, eps_p_2d)
                
                # Setup a fresh _cuMPM.EELS_Solver for this split system
                box_list = [float(x) for x in self.box]
                omega_list = [float(o) for o in self.omega]
                
                # Convert split_eps to nested lists
                split_eps_2d = [[complex(x) for x in row] for row in split_eps]
                
                # Map quad_idxs to the split system
                if self.solve_quadrupoles and len(self.quad_idxs) > 0:
                    split_quad_idxs = []
                    dists_2d = np.linalg.norm(positions[:, :2] - e_pos[np.newaxis, :], axis=1)
                    to_split = dists_2d < self.split_dist
                    
                    unsplit_indices = np.where(~to_split)[0].tolist()
                    split_indices = np.where(to_split)[0].tolist()
                    
                    if self.split_method == "cubic":
                        M = len(self._make_cubic_offsets())
                    else:
                        M = len(self._make_fibonacci_offsets())
                        
                    for q_idx in self.quad_idxs:
                        if q_idx in unsplit_indices:
                            new_idx = unsplit_indices.index(q_idx)
                            split_quad_idxs.append(new_idx)
                        elif q_idx in split_indices:
                            pos_in_split_indices = split_indices.index(q_idx)
                            start_idx = len(unsplit_indices) + pos_in_split_indices * M
                            split_quad_idxs.extend(range(start_idx, start_idx + M))
                else:
                    split_quad_idxs = self.quad_idxs
                
                frame_solver = _cuMPM.EELS_Solver(
                    box_list, split_eps_2d, omega_list, float(self.v), split_R.tolist(),
                    float(self.eps_m), float(self.xi), float(self.tol), bool(self.quiet),
                    self.guess_type, self.solver_type, self.field_type, float(self.integration_step),
                    self.solve_quadrupoles, split_quad_idxs
                )
                
                # Run the solver
                x_part = split_pos[:, 0].tolist()
                y_part = split_pos[:, 1].tolist()
                z_part = split_pos[:, 2].tolist()
                frame_solver.compute(e_pos.tolist(), x_part, y_part, z_part)
                
                # Retrieve EELS spectrum and dipoles
                frame_eels = np.squeeze(frame_solver.get_eels())
                frame_dips = np.squeeze(frame_solver.get_dipoles(physical=False))
                
                if frame_eels.ndim == 0:
                    frame_eels = np.array([frame_eels.item()])
                if len(omega_list) == 1:
                    if frame_dips.ndim == 2:
                        frame_dips = frame_dips[np.newaxis, ...]
                    elif frame_dips.ndim == 1:
                        frame_dips = frame_dips[np.newaxis, np.newaxis, ...]
                
                self._eels_results.append(frame_eels)
                self._dips_results.append(frame_dips)
                self._split_poss.append(split_pos)
                if not hasattr(self, '_dips_scale_factors'):
                    self._dips_scale_factors = []
                self._dips_scale_factors.append(self.eps_m * (split_R[0] ** 3))

                if self.solve_quadrupoles:
                    frame_quads = np.squeeze(frame_solver.get_quadrupoles(physical=False))
                    if len(omega_list) == 1:
                        if frame_quads.ndim == 2:
                            frame_quads = frame_quads[np.newaxis, ...]
                        elif frame_quads.ndim == 1:
                            frame_quads = frame_quads[np.newaxis, np.newaxis, ...]
                    self._quads_results.append(frame_quads)
                    if not hasattr(self, '_quads_scale_factors'):
                        self._quads_scale_factors = []
                    self._quads_scale_factors.append(self.eps_m * (split_R[0] ** 4))
        else:
            x_part = positions[:, 0].tolist()
            y_part = positions[:, 1].tolist()
            z_part = positions[:, 2].tolist()
            for epos_idx in range(num_epos):
                if not self.quiet:
                    print(f"  [EELS] Processing impact parameter {epos_idx + 1}/{num_epos}...")
                self._solver.compute(epos[epos_idx].tolist(), x_part, y_part, z_part)

    def get_eels(self):
        """
        Returns the computed EELS probability spectrum.

        Returns
        -------
        eels : numpy.ndarray
            EELS probability spectrum with shape (num_epos, num_wavelengths) (squeezed if single epos)
        """
        if self.split_dist > 0.0 and self.N_split > 1:
            if not self._eels_results:
                return np.array([])
            return np.squeeze(np.array(self._eels_results))
        else:
            return np.squeeze(self._solver.get_eels())

    def get_dipoles(self, physical=True):
        """
        Returns the computed induced dipoles.

        Parameters
        ----------
        physical : bool, optional
            If True, returns values scaled back to physical units. If False, returns dimensionless values. Defaults to True.

        Returns
        -------
        p : numpy.ndarray or list of numpy.ndarray
            Induced dipoles of each particle. If split, returns a list of arrays (one per epos)
            of shape (num_wavelengths, num_split_particles, 3).
            If shapes across all epos are identical, they are stacked into a single array of shape 
            (num_epos, num_wavelengths, num_split_particles, 3).
            Squeezed if single epos.
        """
        if self.split_dist > 0.0 and self.N_split > 1:
            if not self._dips_results:
                return []
            if physical:
                scaled_results = []
                for dips_arr, factor in zip(self._dips_results, self._dips_scale_factors):
                    scaled_results.append(dips_arr * factor)
                if len(scaled_results) == 1:
                    return np.squeeze(scaled_results[0])
                try:
                    stacked = np.stack(scaled_results)
                    return np.squeeze(stacked)
                except ValueError:
                    return [np.squeeze(d) for d in scaled_results]
            else:
                if len(self._dips_results) == 1:
                    return np.squeeze(self._dips_results[0])
                try:
                    stacked = np.stack(self._dips_results)
                    return np.squeeze(stacked)
                except ValueError:
                    return [np.squeeze(d) for d in self._dips_results]
        else:
            return np.squeeze(self._solver.get_dipoles(physical))

    def get_quadrupoles(self, physical=True):
        """
        Returns the computed induced quadrupoles.

        Parameters
        ----------
        physical : bool, optional
            If True, returns values scaled back to physical units. If False, returns dimensionless values. Defaults to True.

        Returns
        -------
        Q : numpy.ndarray or list of numpy.ndarray
            Induced quadrupoles of each particle. If split, returns a list of arrays (one per epos)
            of shape (num_wavelengths, num_split_particles, 5).
            If shapes across all epos are identical, they are stacked into a single array of shape 
            (num_epos, num_wavelengths, num_split_particles, 5).
            Squeezed if single epos.
        """
        if self.split_dist > 0.0 and self.N_split > 1:
            if not self._quads_results:
                return []
            if physical:
                scaled_results = []
                for quads_arr, factor in zip(self._quads_results, self._quads_scale_factors):
                    scaled_results.append(quads_arr * factor)
                if len(scaled_results) == 1:
                    return np.squeeze(scaled_results[0])
                try:
                    stacked = np.stack(scaled_results)
                    return np.squeeze(stacked)
                except ValueError:
                    return [np.squeeze(q) for q in scaled_results]
            else:
                if len(self._quads_results) == 1:
                    return np.squeeze(self._quads_results[0])
                try:
                    stacked = np.stack(self._quads_results)
                    return np.squeeze(stacked)
                except ValueError:
                    return [np.squeeze(q) for q in self._quads_results]
        else:
            return np.squeeze(self._solver.get_quadrupoles(physical))

    def get_positions(self):
        """
        Returns the particle positions used in the calculation.
        If splitting was enabled, returns a list of positions arrays (one per epos)
        or a single stacked/squeezed array.
        """
        if self.split_dist > 0.0 and self.N_split > 1:
            if not self._split_poss:
                return []
            if len(self._split_poss) == 1:
                return np.squeeze(self._split_poss[0])
            try:
                stacked = np.stack(self._split_poss)
                return np.squeeze(stacked)
            except ValueError:
                return [np.squeeze(p) for p in self._split_poss]
        else:
            return self.positions
