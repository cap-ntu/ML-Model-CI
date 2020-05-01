#!/bin/bash

docker run -d --rm --gpus '"device=1"' -p "${2}":8500 -p "${3}":8501 \
  --mount type=bind,source="${HOME}"/.hysia/"${1}"/tensorflow-tfs,target=/models/"${1}" \
  -e MODEL_NAME="${1}" -t tensorflow/serving
