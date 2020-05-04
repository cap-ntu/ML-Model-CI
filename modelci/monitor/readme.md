# Monitor

## Node Exporter for GPU monitor

start the nvidia node exporter

```bash
sh start_node_exporter.sh
```

check idle GPU ids

```python
from modelci.monitor.gpu_node_exporter import GPUNodeExporter

a = GPUNodeExporter()
a.get_idle_gpu()
# output [0， 1， 2]

```

## cAdvisor for monitoring the other resource usage