# Auto-deployment

## Usage

```shell script
python serving --model {MODEL_NAME} --framework {FRAMEWORK_NAME}
```

Supported model name:

- ResNet50

Supported production model formats associated with serving systems:

- TorchScript -> Self-defined gRPC docker
- TensorFlow SavedModel -> Tensorflow-Serving
- ONNX -> ONNX runtime
- TensorRT -> TensorRT inference Server (can also support all above formats)

Support production communication protocal

- HTTP
- gRPC
