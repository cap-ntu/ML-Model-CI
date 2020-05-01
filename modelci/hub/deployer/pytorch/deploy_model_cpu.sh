#!/bin/bash

docker run -d --rm -p "${2}":8000 -p "${3}":8001 \
  --mount type=bind,source="${HOME}"/.hysia/"${1}"/pytorch-torchscript,target=/model/"${1}" \
  -e MODEL_NAME="${1}" -t pytorch-serving
