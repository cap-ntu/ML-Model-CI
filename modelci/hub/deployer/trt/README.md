# TensorRT serving

## Quick startup

We assume you have setup the MongoDB and all environment variables by [install](/README.md#installation).
See `modelci/hub/init_data.py`.  
Converted TRT model from TensorFlow ResNet50.
```shell script
python modelci/init_data.py --model resnet50 --framework tensorflow --trt
```
Models will be saved at `~/.modelci/ResNet50/tensorflow-trt/` directory.  


Deploy with Triton Docker:
```shell script
sh deploy_model.sh {MODEL_NAME} {REST_API_PORT} {GRPC_PORT} {METRICS_PORT}
```
For example:
```shell script
sh deploy_model.sh ResNet50 8200 8201 8202
```

## Testing

From [TensorRT Client](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/client.html#getting-the-client-libraries).  

### Test
```shell script
python client.py -m ResNet50 -a -u localhost:8201 -s VGG -i grpc -b 8 img_bigbang_scene.jpg
```

### (Optional) Regenerate Type Hint
Triton client (tensorrtserver) uses a generated python version proto buffer for gRPC call. The issues lines that the
generated code has no readability and unfriendly to IDE which has a functionality of type hint. Therefore, we 
generate data objects as utility objects by [betterproto](https://github.com/danielgtaylor/python-betterproto).  
 
We provide data objects generated from tensorrt-inference-server v19.10. If you are dealing with some compatible issues
with your Triton server library version, you can regenerate the data objects by:    
```shell script
VERSION=19.10  # change to your version
FILENAME=r"${VERSION}".zip
LIB_DIR="tensorrt-inference-server-${VERSION}"

TRT_BASE_DIR="${PWD}"

wget https://github.com/NVIDIA/tensorrt-inference-server/archive/"${FILENAME}"
unzip "${FILENAME}"

cd "${LIB_DIR}"/src
# You are to modify `core/grpc_service.proto` by hand, comment service GRPCService from line 42 to line 89:
# 42 | service GRPCService
# 43 | {
#  :
# 85 |    rpc SharedMemoryControl(SharedMemoryControlRequest)
# 86 |       returns (SharedMemoryControlResponse)
# 87 |   {
# 88 |   }
# 89 | }

# generate better protobuf
python -m grpc_tools.protoc -I core --python_betterproto_out=. \
  core/grpc_service.proto core/grpc_service.proto core/model_config.proto \
  core/request_status.proto core/server_status.proto

# the generate code will be at `nvidia/inferenceserver.py`
mv nvidia/inferenceserver.py "${TRT_BASE_DIR}"

# clean
rm "${FILENAME}"
rm -r "${LIB_DIR}"
```
You are required to install additional libraries. You may want to create a new virtual environment before you 
regenerate:
```shell script
conda env create -f gen-hint-environment.yml
```
Or:
```shell script
conda install -y grpcio grpcio-tools
pip install 'betterproto[compiler]'
```

## Workflow

1. Get model in other serving engine (TFS, plain TF, ONNX...)
2. Convert to TensorRT using some library (
[tensorrt](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_onnx_python), 
[onnx-tensorrt](https://github.com/onnx/onnx-tensorrt), ...)
    1. ONNX to TRT
        ONNX can convert to a trt model: `modal.plan`. However, some ONNX ops are not supported by TRT. See 
        supported ONNX version for specific TRT version [here](#trt-compatibility).
    2. TF 2.0 to TRT  
        TF 2.0 does not support converting frozen graphs. We are using SavedModel to TF-TRT serving.
3. Generate model configuration  
    [Model Configuration sample](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html#section-model-configuration)
    ```text
    name: "my_model"
    platform: "tensorrt_plan"
    max_batch_size: 8
    input [
      {
        name: "input0"
        data_type: TYPE_FP32
        dims: [ 16 ]
      },
      {
        name: "input1"
        data_type: TYPE_FP32
        dims: [ 16 ]
      }
    ]
    output [
      {
        name: "output0"
        data_type: TYPE_FP32
        dims: [ 16 ]
      }
    ]
    ```
    `dim` may have some issues: 
    See [Triton#998](https://github.com/NVIDIA/tensorrt-inference-server/issues/998)
4. Store the model in a inferencable directory format
    [Model Repository](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_repository.html)
    - TRT Model:
        ```text
        models/
          <model-name>/
            config.pbtxt
            <version>/
              model.plan
        ```
    - TF SavedModel
        ```text
        models/
          <model-name>/
            config.pbtxt
            <version>/
              model.savedmodel/
                <saved-model files>
        ```
5. Start Inference Server Docker
    ```shell script
    sh deploy_model.sh {MODEL_NAME} {REST_API_PORT} {GRPC_PORT} {METRICS_PORT}
    ```
    `--ulimit` is used for display and setting usage limit in the Docker container. See 
    [Set ulimits in container](https://docs.docker.com/engine/reference/commandline/run/#set-ulimits-in-container---ulimit).

6. (Optional) Checking running status
    ```shell script
    curl localhsot:8000/api/status
    ```

## Optimization

## TRT Compatibility
TensorRT version and dependent libraries version compatibility:

| <small>TensorRT | CUDA                           | cuDNN | Supported ONNX   | PyTorch | TensorFlow </small> |
|:---------------:|:-------------------------------|:------|:-----------------|:--------|:--------------------|
| 7.0.0           | 9.0, 10.0, 10.2                | 7.6.5 | 1.6.0 (opset 10) | 1.3.0   | 1.14.0              |
| 6.0.1           | 9.0, 10.0, 10.1 update 2, 10.2 | 7.6.5 | 1.5.0 (opset 9)  | 1.1.0   | 1.14.0              |
