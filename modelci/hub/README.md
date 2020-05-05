# ModelHub

Manage (CURD), convert, diagnose and deploy DL models supported by industrial serving systems.

## Manage

A collection of high level APIs to drive model service, including register (including uploading) models with a branch of auto-generated model family, select suitable model based on requirement and other model management APIs.

## Convert

To help developers convert models for deployment purpose

- [x] Pytorch -> torchscript
- [x] Pytorch -> ONNX
- [x] Tensorflow -> Tensorflow-Serving format
- [x] Tensorflow -> TensorRT format
- [x] ONNX -> TensorRT format

## Diagnose

Test model at production environment

- [ ] Get original model performance (cold start, latency, throughput) on different devices (Local)
- [ ] Get containerized model performance (cold start, latency, throughput) on different devices on different devices (container)
- [ ] Successful converted models with their performance and failed converted models with error codes
- [ ] Accuracy (or some other metrics) loss of new converted models (need users to specify the path of test data)

## Deploy

Employ serving systems and docker to deploy models

- [x] Torchscript
- [x] Tensorflow-Serving
- [x] ONNX_Runtime
- [x] TensorRT-Inference-Server
