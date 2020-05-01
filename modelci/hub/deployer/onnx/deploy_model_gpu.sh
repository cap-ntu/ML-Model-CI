#!/bin/bash

docker run -d --rm -p "${2}":8000 \
  --gpus all \
  --mount type=bind,source="${HOME}"/.hysia/"${1}"/pytorch-onnx,target=/model/"${1}" \
  -e MODEL_NAME="${1}" -t onnx-serving-gpu
