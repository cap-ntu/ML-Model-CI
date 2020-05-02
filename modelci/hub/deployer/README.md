# Auto-deployment

## Usage

```shell script
python serving --model {MODEL_NAME} --framework {FRAMEWORK_NAME}
```

Supported model name:

- ResNet50

Supported production model formats associated with serving systems:

- TorchScript -> Self-defined gRPC docker
- TensorFlow SaveModel -> Tensorflow-Serving
- ONNX -> ONNX runtime
- TensorRT -> TensorRT inference Server

Support production communication protocal

- RESTful
- gRPC
