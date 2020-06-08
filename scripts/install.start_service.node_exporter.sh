#!/usr/bin/env bash

docker run -d --runtime=nvidia --rm \
  --name=modelci.dcgm-exporter \
  bgbiao/dcgm-exporter
docker run -d --privileged --rm \
  -p 9400:9400 \
  --volumes-from modelci.dcgm-exporter:ro \
  --name modelci.gpu-metrics-exporter \
  bgbiao/gpu-metrics-exporter
