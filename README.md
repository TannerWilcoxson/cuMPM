# cuMPM: CUDA-Accelerated Dipole Solver

cuMPM is a high-performance, CUDA-accelerated implementation of the Dipole Solver for calculating particle dipoles and polarizabilities under periodic boundary conditions using 3D Ewald summation and a GPU-resident restarted complex GMRES solver.

It is structured as an installable Python package utilizing `pybind11` and `scikit-build-core` for seamless compilation and integration with NumPy.

---

## Prerequisites

To install and compile cuMPM, you need:
- **CUDA Toolkit** (11.0+ recommended, includes `nvcc` and `cufft`)
- **CMake** (v3.18 or higher)
- **C++ Compiler** supporting C++17 (e.g., GCC 9+)
- **Python Environment** (>= 3.8, with `pip` and `numpy`)

---

## Installation

You can install the package directly using `pip`. 

Navigate to the project root directory and run:

### Standard Installation
```bash
pip install .
```

### Editable Installation (for development)
```bash
pip install -e .
```

During installation, the build backend (`scikit-build-core`) will automatically invoke CMake, detect your CUDA toolkit/compiler, build the high-performance C++ module, and package it next to the Python wrapper.

---

## Python Interface Usage

Once installed, you can import and use `cuMPM` in any Python environment:

```python
import numpy as np
import cuMPM

# Define lattice box size [L_x, L_y, L_z] in nm
box = [50.0, 50.0, 1000.0]

# Define complex particle polarizabilities (spectral points)
eps_p = [complex(3.0, 0.1), complex(3.2, 0.15)]

# Define coordinates for N particles
positions = np.array([
    [0.0, 0.0, 0.0],
    [10.0, 10.0, 0.0]
])

# Initialize the solver
solver = cuMPM.dipole_solver(box=box, eps_p=eps_p, radius=5.0, eps_m=1.0)

# Compute dipoles
solver.compute(positions)

# Extract results
alpha_eff = solver.get_eff_polarizability()
p = solver.get_dipoles()
```

### Running Simulations

To run the hexatic lattice simulation case study (incorporating a frequency-dependent Drude model for ITO):
```bash
python drude_ito_simulation.py
```
This runs the simulation and generates `drude_ito_results.png` showing the plasmon resonance spectrum.

---

## Directory Structure
- `src/cuMPM/`: Contains the user-facing Python package (`__init__.py`).
- `src/cpp/`: Core C++/CUDA source files:
  - `dipole_solver_bindings.cpp`: `pybind11` module definition and mappings.
  - `dipole_solver.cu` / `.h`: GPU-resident complex GMRES solver.
  - `electric_field.cu` / `.h`: 3D Ewald summation grid-spreading and field calculation.
- `CMakeLists.txt`: Configures compilation targets.
- `pyproject.toml`: Modern Python packaging and build configuration.
