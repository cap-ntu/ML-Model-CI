# Dispatch a Model as a Cloud Service (MLaaS)

After the profiling, you understand your model's runtime performance. With this performance as a guideline, you can dispatch your model as an efficient cloud service with MLModelCI's dispatch API.

Before you start trying these features, please make sure you have installed the MLModelCI correctly and started the mongodb service as well. You can refer to the [installation](../../README.md#installation) for more details.

## Deploy a Converted and Profiled Model

Our dispatch supports to load models with the following serving systems:

- Triton Inference System
- TensorFlow-Serving
- ONNX Runtime
- Self-defined TorchScript Container

### Server Installation

Before serving your models, please make sure you have installed the Docker images of above serving systems. By default, the MLModelCI's Docker image contains them, so you don't need to install them again.

**MLModelCI/PyTorch-Serving**

![](https://img.shields.io/docker/pulls/mlmodelci/pytorch-serving.svg) ![](https://img.shields.io/docker/image-size/mlmodelci/pytorch-serving)

```bash
docker pull mlmodelci/pytorch-serving
```

**MLModelCI/ONNX-Serving**

![](https://img.shields.io/docker/pulls/mlmodelci/onnx-serving.svg) ![](https://img.shields.io/docker/image-size/mlmodelci/onnx-serving)

```bash
docker pull mlmodelci/onnx-serving
```

**Triton Serving System**

```bash
docker pull nvcr.io/nvidia/tensorrtserver:19.10-py3
```

**TensorFlow-Serving**

```bash
docker pull tensorflow/serving
```

### Dispatch API

The dispatch API launches a serving system which loads a model and run it in a containerized manner.


You can get the model path by using `retrieve` API and it will return a `saved_path` (See [Tricks with Model Saved Path](./register.md#tricks-with-model-saved-path)) to specify model local cache. 

Now you can assign a device (i.e. `'cpu'`, `'cuda:0'`, `'cuda:0,1'`) and a batch size to serve the model with the profiling results as a guideline.

Or MLModelCI will set them automatically. 


```python
from modelci.hub.deployer import serve

saved_path = ...
device = '1'
batch_size = 8
server_name = 'name of container'

serve(save_path=saved_path, device=f'cuda:{device}', name=server_name, batch_size=batch_size)
```

If you want to stop the running container, you can simply stop service in your terminal.

```bash
docker stop <name>
```

The model will be removed once stopped.
