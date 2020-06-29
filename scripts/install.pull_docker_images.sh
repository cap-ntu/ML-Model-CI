#!/usr/bin/env bash

# Get log path
NOW=$(date +'%Y%m%d-%H%M%S')
LOG_PATH=${1:-/tmp/modelci-install-${NOW}.log}

images=(
  'mongo'
  'bgbiao/dcgm-exporter'
  'bgbiao/gpu-metrics-exporter'
  'gcr.io/google-containers/cadvisor'
  'mlmodelci/pytorch-serving:latest'
  'mlmodelci/pytorch-serving:latest-gpu'
  'mlmodelci/onnx-serving:latest'
  'mlmodelci/onnx-serving:latest-gpu'
  'tensorflow/serving:2.1.0'
  'tensorflow/serving:2.1.0-gpu'
  'nvcr.io/nvidia/tensorrtserver:19.10-py3'
)

for image in "${images[@]}";
do
  docker pull -q "${image}" >> "${LOG_PATH}"
  printf '.'
done
