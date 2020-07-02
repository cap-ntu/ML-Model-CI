#!/usr/bin/env bash
# Install Conda environment
conda env create -f environment.yml

# Activate conda
source "${CONDA_PREFIX}"/etc/profile.d/conda.sh
conda activate modelci

# Install PyTorch and TensorFlow package
conda install pytorch=1.5.0 torchvision cpuonly -y -c pytorch
conda install tensorflow=2.1.0 -y
pip install tensorflow-serving-api==2.1.0
