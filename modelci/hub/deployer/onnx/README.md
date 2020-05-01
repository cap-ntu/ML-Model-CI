# ONNX Serving

Road map  
- [x] try official script for deploying ONNX model via a REST API with FastAPI  
- [x] serve Resnet50  
- [x] pack a ONNX serving docker  
- [ ] add gRPC support of the ONNX serving  
- [ ] API test script and gRPC test script  
- [ ] API and gRPC test with profiling  

## Install
```shell script
cp ../config/imagenet_class_index.json .

# Build Docker
docker build -t pytorch-serving -f onnx-serve-cpu.Dockerfile .  
# For GPU version, use onnx-serve-gpu.Dockerfile instead
```

## Usage
1. Get pretrained torch model
    See `modelci/hub/init_data.py`.
    The model will be saved at `~/.hysia/ResNet50/pytorch-onnx/` directory.

2. deploy model
    CPU version:
    ```shell script
    sh deploy_model_cpu.sh {MODEL_NAME} {REST_API_PORT}
    ```
    GPU version:
    ```shll script
    sh depolu_model_gpu.sh {MODEL_NAME} {REST_API_PORT}
    ```
