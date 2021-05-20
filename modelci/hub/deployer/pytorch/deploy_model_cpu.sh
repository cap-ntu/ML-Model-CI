#!/bin/bash

docker run -d --rm -p "${2}":8000 -p "${3}":8001 \
  --mount type=bind,source="${HOME}"/.modelci/"${1}"/pytorch-torchscript,target=/models/"${1}" \
  -e MODEL_NAME="${1}" --env-file docker-env.env -t mlmodelci/pytorch-serving:latest