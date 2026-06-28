#!/bin/bash
set -e

# Clean path to remove Windows path sharing parentheses
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '(' | tr '\n' ':')

# Prepend CUDA path
export PATH=/usr/local/cuda/bin:$PATH

cd /home/tanner/cuMPM

# Clear old build directories if --clean or -c is passed
if [ "$1" == "--clean" ] || [ "$1" == "-c" ]; then
    echo "Cleaning build directories..."
    rm -rf build _skbuild .pybuild src/cuMPM.egg-info
fi

# Rebuild and install (verbose)
/home/tanner/micromamba/envs/cuMPM/bin/pip install -v -e . --no-build-isolation

# Run tests
/home/tanner/micromamba/envs/cuMPM/bin/python -m pytest tests/ -v
