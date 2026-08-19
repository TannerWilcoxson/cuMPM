# cuMPM: CUDA-Accelerated Dipole Solver

cuMPM is a high-performance, CUDA-accelerated implementation of the Dipole Solver for calculating particle dipoles and polarizabilities under periodic boundary conditions using 3D Ewald summation and GPU-resident complex numerical solver solvers.

It is structured as an installable Python package utilizing `pybind11` and `scikit-build-core` for seamless compilation and integration with NumPy.

---


- **CUDA Toolkit** (11.0+ recommended, includes `nvcc` and `cufft`)
- **CMake** (v3.18 or higher)
- **C++ Compiler** supporting C++17 (e.g., GCC 9+)
- **Python Environment** (>= 3.8, with `pip` and `numpy`)

---

## Installation

### Conda/Mamba Environment Installation (Recommended)
You can set up a complete development environment (including all compilation, testing, and documentation tools) using Mamba or Conda with the provided `environment.yml` file:

```bash
# Create the environment
mamba env create -f environment.yml

# Activate the environment
mamba activate cuMPM

# Install the package
pip install .
```

### Standard Installation (Pip Only)
You can also install the package directly using `pip`. Navigate to the project root directory and run:

```bash
pip install .
```

During installation, the build backend (`scikit-build-core`) will automatically invoke CMake, detect your CUDA toolkit/compiler, build the high-performance C++ module, and package it next to the Python wrapper.

---

## Python Interface Usage

Once installed, you can import and run `cuMPM` within Python scripts. 

cuMPM provides:
1. **`dipole_solver`**: Solves self-consistent dipoles (and optionally quadrupoles) of particle arrays under incident plane wave fields with periodic/open boundary conditions.
2. **`Near_Field`**: Computes local field intensity at arbitrary spatial coordinates.
3. **`EELS`**: Computes Electron Energy Loss Spectroscopy (EELS) probability spectra and induced dipoles/quadrupoles along the trajectory of a relativistic electron beam.

For complete, executable simulation setups (such as Drude ITO hexatic lattice modeling, periodic EELS with quadrupoles, and mixed permittivity simulations), please refer to the `examples/` directory.

---

## Performance & Benchmarks

### 1. GPU Acceleration vs CPU Baseline ($N = 4,000$ Particles)

Evaluating self-consistent dipoles across 10 frequency points for a 2D hexagonal monolayer ($N = 4,000$ particles):

| Implementation | Total Time (10 Freqs) | Time / Frequency | Speedup |
| :--- | :---: | :---: | :---: |
| **[pyMPM](https://github.com/truskett-group-ut/pyMPM)** (CPU) | 75.10 s | 7,510 ms | 1.0x |
| **cuMPM** (GPU - BiCGSTAB) | **1.36 s** | **136 ms** | **55.4x** |
| **cuMPM** (GPU - GMRES) | **1.32 s** | **132 ms** | **56.9x** |

*Benchmark script available at `benchmarks/benchmark_cumpm_vs_pympm.py`.*

### 2. Jacobi Preconditioning Speedup ($N = 1,000$ Polydisperse Particles)

Convergence speedup on polydisperse clusters with mixed radii ($1.0\text{ nm}$ to $4.0\text{ nm}$) across 20 frequency steps (`tol = 1e-6`):

| Solver Mode | Total Execution Time | Time / Frequency | Speedup / Time Saved |
| :--- | :---: | :---: | :---: |
| **Jacobi Preconditioning OFF** | 103.17 s | 5,158 ms | Baseline |
| **Jacobi Preconditioning ON** | **7.50 s** | **375 ms** | **13.76x Faster (+92.73% time saved)** |

*Benchmark script available at `benchmarks/benchmark_jacobi.py`.*

---

## Documentation

The Sphinx HTML documentation is precompiled and included directly in this repository. To view the documentation, open `cuMPM/docs/build/html/index.html` in your web browser. 

---

## Directory Structure

* **`src/cuMPM/`**: User-facing Python package containing wrappers:
  * `solver.py`: Wrapper for the `dipole_solver` interface.
  * `near_field.py`: Wrapper for the `Near_Field` field evaluator.
* **`src/cpp/`**: Core high-performance C++/CUDA source files:
  * `dipole_solver.cu` / `.h`: Resolute GMRES solver and system configurations.
  * `near_field.cu` / `.h`: Electric field intensity calculation.
  * `dipole_solver_bindings.cpp`: Pybind11 registration and NumPy mapper helpers for solver classes.
  * `near_field_bindings.cpp`: Pybind11 mappings for the `Near_Field` evaluator.
  * `electric_field/`: Ewald and Polydisperse CUDA electric field summation solvers.
  * `numerical_solver/`: Krylov subspace solvers (restarted GMRES and BiCGSTAB).
* **`docs/`**: Sphinx documentation configuration, source files, and precompiled HTML files (`docs/build/html/`).
* **`tests/`**: Modular test suite verified by `pytest`.
* **`examples/`**: Case study and executable demo setups (e.g. Drude-ITO lattice modeling and 3D polydisperse NaCl lattices).
* **`benchmarks/`**: Scripts to compare solver runtimes and verify scaling on GPU.
* **`environment.yml`**: Portable environment setup file for Conda/Mamba.
* **`CMakeLists.txt`**: C++ CMake compilation pipeline configurations.
* **`pyproject.toml`**: Package metadata, setup tools, optional-dependencies, and environment configurations.
