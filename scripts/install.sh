#!/usr/bin/env sh

now=$(date +'%Y%m%d-%H%M%S')

# Install Conda environment
echo "Installing Conda environment..."
conda env create -f environment.yml

# Install TRTIS client APIs
echo "Installing TRTIS client API"
sh scripts/install_trtis_client.sh

# Start MongoDB service
echo "Starting services..."
sh scripts/start_service.sh
