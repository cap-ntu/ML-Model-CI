# CAdvisor for Model CI

## Quick Start 

### Start cAdvisor

First, start cAdvisor Docker container:

```shell script
sh <project-root>/script/start_advisor.sh 
```

if you want to add support for GPU devices:

```shell script
sh <project-root>/script/start_advisor.sh -gpu
```

You can use `-h` or `--help` to get more information about this script.

### Request Information

Using following code, you can get the required formated data from [cAdvisor](https://github.com/google/cadvisor/).

```python
 
from cadvisor import CAdvisor

client = CAdvisor(base_link='') # link to your remote 
tensorflow_resnet_latest = client.request_by_image('bitnami/tensorflow-serving:latest') # set filters
out = client.get_model_info(tensorflow_resnet_latest) # request JSON data
```
[Here](./example_data/example_tf_resnet.json.json) is an example of required JSON data.

For detailed information about setting filters, please refer to the [client document](./CLIENT.md).

## Data Format 

Here is an example of data structure:

```json
{
    "/docker/29c1743de5a87951dc7689780d40399843fc790efe28327f34eb20710668b661":{
        "id":"29c1743de5a87951dc7689780d40399843fc790efe28327f34eb20710668b661",
        "creation_time":"2020-04-21T11:51:29.331285325Z",
        "image_name":"bitnami/tensorflow-serving:latest",
        "has_cpu":true,
        "cpu":{
            "limit":1024,
            "max_limit":0,
            "mask":"0-27",
            "period":100000
        },
        "has_memory":true,
        "memory":{
            "limit":9223372036854771712,
            "reservation":9223372036854771712
        },
        "stats":[
            Object{...},
            Object{...},
            {
                "timestamp":"2020-04-23T04:00:12.928166614Z",
                "cpu":Object{...},
                "memory":Object{...},
                "accelerators":Array[1]
            },
            ...
}
```

Note, `stats` is a list includes every second status information up to last 60 seconds.

## About GPU Accelerator 

- current cAdvisor DOES NOT support presenting accelerator data at the web UI
- current cAdvisor has limited metrics
- current cAdvisor can only support Nvidia GPUs.

A sample accelerator dict looks like:

```json
{
"accelerators": [{
		"make": "nvidia",
		"model": "GeForce RTX 2080 Ti",
		"id": "GPU-771dd245-2cab-c7c3-8460-6bfd4d4b5659",
		"memory_total": 11554717696,
		"memory_used": 1048576,
		"duty_cycle": 0
	}, {
		"make": "nvidia",
		"model": "GeForce RTX 2080 Ti",
		"id": "GPU-0daeb0b1-fa19-7218-9810-dc9ce9d5ac8d",
		"memory_total": 11554717696,
		"memory_used": 1048576,
		"duty_cycle": 0
	}, {
		"make": "nvidia",
		"model": "GeForce RTX 2080 Ti",
		"id": "GPU-80464f50-64c5-687a-0b34-8ea7cd326ac2",
		"memory_total": 11554717696,
		"memory_used": 1048576,
		"duty_cycle": 0
	}, {
		"make": "nvidia",
		"model": "GeForce RTX 2080 Ti",
		"id": "GPU-60ece064-e9e8-fd74-9380-07de8abdfb4c",
		"memory_total": 11551440896,
		"memory_used": 64094208,
		"duty_cycle": 0
	}]
}
```

if you want more metrics about GPU, please refer to [here](https://github.com/google/cadvisor/issues/2271).

## Roadmap

- [x] Improve the data format and API of Python client
- [ ] Integrate with locally collector

