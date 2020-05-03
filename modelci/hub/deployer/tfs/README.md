# TensorFlow Serving
Deploy Keras model with TensorFlow Serving.  
CPU version is based on the latest tensorflow version (2.0.0).
GPU version is not supported yet due to some TF/CUDA version issues.

## Usage

<ol>
<li> Get pre-trained Keras model

We assume you have setup the MongoDB and all environment variables [install](/README.md#installation). 
See `modelci/hub/init_data.py`.  
For example, running 
```shell script
python modelci/init_data.py --model resnet50 --framework tensorflow
```
Models will be saved at `~/.modelci/ResNet50/tensorflow-tfs/` directory.  
**Note**: You do not need to rerun the above code if you have done so for [TensorRT](/modelci/hub/deployer/trt).

</li>
<li> Deploy model

```shell script
sh deploy_model_cpu.sh {MODEL_NAME} {GRPC_PORT} {REST_API_PORT}
```
Or on a gpu
```shell script
sh deploy_model_gpu.sh {MODEL_NAME} {GRPC_PORT} {REST_API_PORT}
```
You may check the deployed model using `save_model_cli` from https://www.tensorflow.org/guide/saved_model
```shell script
saved_model_cli show --dir {PATH_TO_SAVED_MODEL}/{MODEL_NAME}/{MODEL_VARIANT}/{MODEL_VERSION} --all
```

</li>
<li> Testing

```shell script
# 4.1. MAKE REST REQUEST
python rest_request.py --model {MODEL_NAME} --port {PORT}
# 4.2. MAKE GRPC REQUEST
python grpc_request.py --model {MODEL_NAME} --input_name {INPUT_NAME}
```

</li>
</ol>

## Example
Let's deploy a pre-trained ResNet50 model (at working directory `modelci/hub`)
```shell script
python init_data.py export --model resnet50 --framework tensorflow
bash deploy_model_gpu.sh resnet50 8500 8501
saved_model_cli show --dir ./resnet50/1 --all
python rest_client.py --model resnet50 --port 8501
python grpc_client.py --model resnet50 --input_name input_1

# FOR TESTING THE LATENCY AND THROUGHPUT
python rest_client.py --model resnet50 --port 8501 -t
python grpc_client.py --model resnet50 --input_name input_1 -t
```

