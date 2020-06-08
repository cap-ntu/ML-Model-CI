#!/usr/bin/env bash

function clean_docker() {
    local name="modelci.test._*"
    # shellcheck disable=SC2046
    docker stop $(docker ps -a -q filter="name=${name}")
}

source "${HOME}"/anaconda3/etc/profile.d/conda.sh
conda activate modelci

source scripts/setup_env.sh
export MONGO_DB='test'

python -m pytest tests/

cd modelci/hub || exit 1
python init_data.py export --model ResNet50 --framework tensorflow
python init_data.py export --model ResNet50 --framework pytorch

cd deployer || exit 1
# test ts
python serving.py name -m ResNet50 -f pytorch -e torchscript --device cpu --name modelci.test._resnet50-ts
python serving.py name -m ResNet50 -f pytorch -e torchscript --device cuda --name modelci.test._resnet50-ts-gpu
# TODO: client
clean_docker modelci.test.resnet50-ts*

# test tfs
python serving.py name -m ResNet50 -f tensorflow -e tfs --device cpu --name modelci.test._resnet50-tfs
python serving.py name -m ResNet50 -f tensorflow -e tfs --device cuda --name modelci.test._resnet50-tfs-gpu
# TODO: client

# test onnx
python serving.py name -m ResNet50 -f pytorch -e onnx --device cpu --name modelci.test._resnet50-onnx
python serving.py name -m ResNet50 -f pytorch -e onnx --device cuda --name modelci.test._resnet50-ts-gpu
# TODO: client

# test trt
python serving.py name -m ResNet50 -f tensorflow -e trt --device cuda --name modelci.test.resnet50-trt
# TODO: client
