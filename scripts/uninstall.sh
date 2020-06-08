#!/usr/bin/env sh
# remove Conda environment
conda remove --name modelci --all -y

# stop docker service
# shellcheck disable=SC2046
docker stop $(docker ps -a -q --filter="name=modelci.*")
