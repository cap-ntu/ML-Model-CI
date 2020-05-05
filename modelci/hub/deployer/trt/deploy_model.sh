#!/usr/bin/env bash

# TODO: combine TF-TRT and ONNX converted TRT
docker run --gpus=all --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67100864 \
    -p 8000:"${2}" -p 8001:"${3}" -p 8002:"${4}" \
    -v /"${HOME}"/.modelci/"${1}"/tensorflow-trt:/models/"${1}" \
    nvcr.io/nvidia/tensorrtserver:19.10-py3 \
    trtserver --model-repository=/models
