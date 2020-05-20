#!/bin/bash

docker run -ti --rm -p "${2}":8000 -p "${3}":8001 \
  --gpus all \
  --mount type=bind,source="${HOME}"/.modelci/"${1}"/pytorch-onnx,target=/models/"${1}" \
  -e MODEL_NAME="${1}" --env-file docker-env.env -t onnx-serving:latest-gpu '/bin/bash'
