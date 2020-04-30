# CAdvisor Python Client

We use [cAdvisor](https://github.com/google/cadvisor) to monitor the performance of running Docker containers.

## Quick Start

### Start the CAdvisor Service

We use cAdvisor Docker container to monitor the specific Docker containers, before running the cAdvisor Python client, please start the cAdvisor locally by:

```bash
sh ./script/start_advisor.sh 
```

if you want to add support for GPU devices:

```bash
sh ./script/start_advisor.sh -gpu
```

You can use `-h` or `--help` to get more information.

### Usage

#### Init the CAdvisor Client

Init the cAdvisor client, you can set the API informations here.

```python
cAdvisor = CAdvisor() # init with default settings
cAdvisor = CAdvisor(base_link='http://localhost:8080/api/', ersion='v1.2', query_object='docker') 
# init with specific API dettings
```

#### Request the Host Machine Information

You can also get the host machine info by:

```python
machine_info = cAdvisor.machine_info()
```

#### Get Basic Information from Running Containers

If you just want to check the information of running containers, you can do:

```python
data_running_containers = cAdvisor.get_running_container()
# it also receives an input screened dict
all_information_by_id = cAdvisor.request_by_id("afcb380e62bbec4fb7992cda3c986d387b5bc137fdea7dc0d1d13448012d5a5d")
data_running_container_by_id = cAdvisor.get_running_container(all_information_by_id)
```

#### Request All Container Data

Request the JSON data, each request will return the information in the last one minute (60 seconds), if the container just started within 1 min, the request will return all of them.

```python
all_data = cAdvisor.request_all() # request all the data at once
all_stat = cAdvisor.get_stats() # request all the stats per second for all running containers
```

#### Request Data from Specific Containers

You can set filters to the request, if we only need information from containers with image `ubuntu:16.04`:

```python
all_information_ubuntu = cAdvisor.request_by_image("ubuntu:16.04")
```

We can also request data by other fields like name(id or image name), id, or just all the information expect cadvisor itself.

```python
# request by id
all_information_by_id = cAdvisor.request_by_id("afcb380e62bbec4fb7992cda3c986d387b5bc137fdea7dc0d1d13448012d5a5d") 
# request by name or id
all_data_cadvisor = cAdvisor.request_by_name("cadvisor") 
all_data_cadvisor = cAdvisor.request_by_name("afcb380e62bbec4fb7992cda3c986d387b5bc137fdea7dc0d1d13448012d5a5d") 
# request all the container informations without cAdvisor
data_all_without_cadvisor = cAdvisor.request_without_cadvisor()
```

#### Request Data by Specific Metric

If you want to get some specific data by metrics like: diskio, cpu, etc.:

```python
# The field can be: diskio, cpu, diskio, memory, network, filesystem, task_stats, processes, accelerators (Nvidia GPU only)
data_diskio_all = cAdvisor.get_specific_metrics(metric='diskio')
```

It also receives a screened dict if you want the specific metric from some specific containers:

```python
all_information_ubuntu = cAdvisor.request_by_image("ubuntu:16.04")
data_cpu_ubuntu = cAdvisor.get_specific_metrics(input_dict=all_information_ubuntu, metric='cpu')
```

#### Export the Data 

All the data here are Python dicts, you can use json to dump them out if you want to save the data locally.

```python
import json
all_data = cAdvisor.request_all()
json.dumps(all_data)
```

## Example

Here is a [quick start python sample](./sample.py) for the client.

### Start Containers

Here we started several demo containers in our server for an example. 

```bash
$ docker run -ti --gpus all nvidia/cuda:10.1-base bash
$ docker run -ti --gpus '"device=1,2"' nvidia/cuda:10.1-base bash
$ docker run -d -v /home/hyz/workspace/model-data:/bitnami/model-data --name tensorflow-resnet --net tensorflow-tier --gpus '"device=1,2"'  --device /dev/nvidia0:/dev/nvidia0 bitnami/tensorflow-resnet:latest
```

More information about the [tensorflow-resnet](https://hub.docker.com/r/bitnami/tensorflow-resnet).

Currently, we have three containers based on the GPU accelerators:

- nvidia/cuda:10.1-base: all 4 GPUs.
- nvidia/cuda:10.1-base: only 1 and 2 GPU devices.
- bitnami/tensorflow-resnet:latest: only 1 and 2 GPU devices.

### Getting Data

We use below code to get data

```python 
cAdvisor = CAdvisor()
example_cuda_101base = cAdvisor.request_by_image("nvidia/cuda:10.1-base")
with open('./example_data/example_cuda_101base.json', 'w') as f:
    json.dump(example_cuda_101base, f)
```

The same as file `example_tensorflow-resnet.json`, for concrete JSON format, please refer to [here](./example_data). 
