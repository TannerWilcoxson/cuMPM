Usage Guide
===========

Solving for Dipoles
-------------------
The core of cuMPM is the :class:`cuMPM.dipole_solver` class, which calculates the self-consistent induced dipoles of many-particle configurations.

Example:

.. code-block:: python

   import numpy as np
   import cuMPM

   # Define box dimensions [Lx, Ly, Lz] in nm
   box = [50.0, 50.0, 100.0]

   # Complex particle permittivity/dielectric constant
   eps_p = 3.0 + 0.1j

   # Initialize solver for particles of radius 1.0 nm
   solver = cuMPM.dipole_solver(box=box, eps_p=eps_p, radius=1.0)

   # Particle positions
   positions = np.array([
       [0.0, 0.0, 0.0],
       [10.0, 0.0, 0.0]
   ])

   # Compute the dipoles
   solver.compute(positions)

   # Fetch effective polarizability and dipoles
   alpha_eff = solver.get_eff_polarizability()
   dipoles = solver.get_dipoles()

Calculating Near Fields
-----------------------
Once dipoles are determined, you can compute the electric field intensity at arbitrary target field points using :class:`cuMPM.Near_Field`.

Example:

.. code-block:: python

   import numpy as np
   import cuMPM

   box = [50.0, 50.0, 100.0]
   E0 = [1.0, 0.0, 0.0]  # Incident electric field vector
   radius = 1.0

   # Dipoles (from solver or specified directly)
   dipoles = np.array([
       [0.1, 0.0, 0.0],
       [-0.1, 0.0, 0.0]
   ])
   dip_pos = np.array([
       [0.0, 0.0, 0.0],
       [10.0, 0.0, 0.0]
   ])

   # Field evaluation points
   field_points = np.array([
       [5.0, 0.0, 0.0],
       [0.0, 5.0, 0.0]
   ])

   # Initialize Near Field calculator
   nf = cuMPM.Near_Field(
       box=box,
       E0=E0,
       radius=radius,
       dip=dipoles,
       dip_pos=dip_pos,
       field_points=field_points
   )

   # Calculate electric field intensity at target points
   intensity = nf.calculate()
