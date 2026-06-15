Installation
============

Prerequisites
-------------
To compile and install cuMPM, you need:

* **CUDA Toolkit** (11.0+ recommended)
* **CMake** (v3.18 or higher)
* **C++ Compiler** with C++17 support (e.g., GCC 9+)
* **Python** (>= 3.8) with `numpy`

Conda/Mamba Environment Setup (Recommended)
-------------------------------------------
To set up a complete development environment (containing all required dependencies for compiling, running, testing, and documenting `cuMPM`), you can use Conda or Mamba with the provided `environment.yml` file:

.. code-block:: bash

   # Create the environment
   mamba env create -f environment.yml

   # Activate the environment
   mamba activate cuMPM

   # Install the package
   pip install .

Standard Installation (Pip Only)
--------------------------------
To install the package directly via pip, run:

.. code-block:: bash

   pip install .

