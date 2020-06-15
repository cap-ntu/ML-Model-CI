# Auto-deployment

## Example

<ol>

<li> Serve by name

Serve a ResNet50 model built with TensorFlow framework and to be served by TensorRT, on CUDA device 1:
```shell script
python serving.py name --m ResNet50 -f tensorflow -e trt --device cuda:1
```

</li>

<li> Serve by task

Serve a model performing image classification task on CPU
```shell script
python serving.py task --task 'image classification' --device cpu
```

</li>

</ol>

## Usage

### Serve by name

```shell script
python serving.py name --model/-m {MODEL_NAME} --framework/-f {FRAMEWORK_NAME} --engine/-e {ENGINE_NAME} --device {DEVICE}
```

### Server by task

```shell script
python serving.py task --task {TASK_NAME} --device {DEVICE}
```

Supported model name:
-   ResNet50

Supported production model formats associated with serving systems:
-   TorchScript -> Self-defined gRPC docker
-   TensorFlow SavedModel -> Tensorflow-Serving
-   ONNX -> ONNX runtime
-   TensorRT -> TensorRT inference Server (can also support all above formats)

Support production communication protocol
-   HTTP
-   gRPC
