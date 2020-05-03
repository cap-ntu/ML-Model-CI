# PyTorch Serving

Road map  
- [x] try official script for deploying pytorch model via a REST API with FastAPI  
- [x] serve Resnet50  
- [x] pack a pytorch serving docker  
- [x] add gRPC support of the pytorch serving  
- [ ] API test script and gRPC test script  
- [ ] API and gRPC test with profiling  

## Install
```shell script
cp ../config/imagenet_class_index.json .

# Generate gRPC code
python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. proto/service.proto

# Build Docker
docker build -t pytorch-serving -f torch-serve-cpu.Dockerfile .  
# For GPU version, use torch-serve-gpu.Dockerfile instead
```

## Usage
1. Get pretrained torch model
    See `modelci/hub/init_data.py`.
   The model will be saved at `~/.modelci/ResNet50/pytorch-torchscript/` directory.

2. deploy model
    CPU version:
    ```shell script
    sh deploy_model_cpu.sh {MODEL_NAME} {REST_API_PORT}
    ```
    GPU version:
    ```shell script
    sh deploy_model_cpu.sh {MODEL_NAME} {REST_API_PORT}
    ```
