#!/usr/bin/env sh
conda deactivate

# remove Conda environment
conda remove --name modelci --all -y

# stop docker service
docker stop modelci-mongo
