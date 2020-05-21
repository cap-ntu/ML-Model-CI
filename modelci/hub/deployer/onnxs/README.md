# ONNX Serving

Road map  
- [x] try official script for deploying ONNX model via a REST API with FastAPI  
- [x] serve Resnet50  
- [x] pack a ONNX serving docker  
- [x] add gRPC support of the ONNX serving  
- [ ] API test script and gRPC test script  
- [ ] API and gRPC test with profiling  

## Install

```shell script
cp ../config/utils.py .
cp ../config/docker-env.env.example ./docker-env.env

# Generate gRPC code
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/service.proto

# Build Docker
docker build -t onnx-serving:latest -f onnx-serve-cpu.Dockerfile .  
# For GPU version, use onnx-serve-gpu.Dockerfile instead, tag GPU version as onnx-serving:latest-gpu
# docker build -t onnx-serving:latest-gpu -f onnx-serve-gpu.Dockerfile .
```

## Usage

<ol>
<li> Get pretrained torch model

We assume you have setup the MongoDB and all environment variables by [install](/README.md#installation). 
See `modelci/hub/init_data.py`.  
For example, running 
```shell script
python modelci/init_data.py --model resnet50 --framework pytorch
```
Models will be saved at `~/.modelci/ResNet50/pytorch-torchscript/` directory.  
**Note**: You do not need to rerun the above code if you have done so for [TorchScript](/modelci/hub/deployer/pytorch).

</li>
<li> deploy model

CPU version:
```shell script
sh deploy_model_cpu.sh {MODEL_NAME} {REST_API_PORT}
```
GPU version:
```shell script
sh depoly_model_gpu.sh {MODEL_NAME} {REST_API_PORT}
```

</li>
</ol>
