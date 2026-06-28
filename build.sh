#!/bin/bash
set -e
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '(' | tr '\n' ':')
export PATH=/usr/local/cuda/bin:$PATH
cd /home/tanner/cuMPM
/home/tanner/micromamba/envs/cuMPM/bin/pip install -e . --no-build-isolation
