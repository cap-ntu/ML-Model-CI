# Monitor

## Node Exporter for GPU monitor

start the nvidia node exporter

```bash
sh start_node_exporter.sh
```

check the information

```bash
curl -s localhost:9400/gpu/metrics
```

## cAdvisor for monitoring the other resource usage
