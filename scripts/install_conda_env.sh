#!/usr/bin/env bash

CUDA_VERSION=$(nvidia-smi | grep -oP '(?<=CUDA Version:\s)\d+.\d+')

# Install Conda environment
conda env create -f environment.yml

# Activate conda
source "${HOME}"/anaconda3/etc/profile.d/conda.sh
conda activate modelci

# Install PyTorch and TensorFlow package
conda install pytorch=1.5.0 torchvision cudatoolkit="${CUDA_VERSION}" -y -c pytorch
conda install tensorflow-gpu=2.1.0 -y
pip install tensorflow-serving-api==2.1.0
