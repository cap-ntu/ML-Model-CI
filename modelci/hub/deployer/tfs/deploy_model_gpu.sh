#!/bin/bash

docker run -d --rm -p "${2}":8500 -p "${3}":8501 \
  --mount type=bind,source="${HOME}"/.modelci/"${1}"/tensorflow-tfs,target=/models/"${1}" \
  -e MODEL_NAME="${1}" -t tensorflow/serving:latest-gpu