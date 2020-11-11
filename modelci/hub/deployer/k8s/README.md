# Generate deployment file of cloud service in k8s

## Example

<ol>

<li> Generate a config file for necessary environment variables in deployment file.

```
[remote_storage]
# configuration for pulling models from cloud storage.
storage_type = S3
aws_access_key_id = sample-id
aws_secret_access_key = sample-key
bucket_name = sample-bucket
remote_model_path = models/bidaf-9

[model]
# local model path for storing model after pulling it from cloud
local_model_dir = /models
local_model_name = bidaf-9

[deployment]
# deployment detailed configuration
name = sample-deployment
namespace = default
replicas = 1
engine = ONNX
device = cpu
batch_size = 16
```

`[remote_storage]` defines variables for pulling model from cloud storage. Currently only s3 bucket is supported.

`[model]` defines variables of model path in containers

`[deployment]` defines variables for serving the model as a cloud service

</li>

<li> Generate deployment file to desired output path.

```
from modelci.hub.deployer.k8s.dispatcher import render

render(
    configuration='example/sample_k8s_deployment.conf',
    output_file_path='example/output.yaml'
)
```

</li>

<li> Deploy the service into your k8s cluster

</li>
</ol>


## Usage

The function is for quickly generate deployment file of cloud service with the modelci-compiled model.
We assume you:
- Push the your compiled model to your remote storage
- Have the k8s cluster for deploying the cloud service

