#!/bin/bash

docker run -d --runtime=nvidia --rm --name=nvidia-dcgm-exporter bgbiao/dcgm-exporter

docker run -d --privileged --rm -p 9400:9400  --volumes-from nvidia-dcgm-exporter:ro bgbiao/gpu-metrics-exporter