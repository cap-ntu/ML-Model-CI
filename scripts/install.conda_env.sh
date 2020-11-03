#!/usr/bin/env bash
# Install Conda environment
conda env create -f environment.yml

# Activate conda
source "${CONDA_PREFIX}"/etc/profile.d/conda.sh
conda activate modelci
