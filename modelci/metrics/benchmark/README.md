# Testing Tool for Throughput and Latency

## Metrics

By running once, you can get some performance informations.

- a latency list contains every batch's latency.
- a through list contains every batch's throughput.
- overall latency
- overall throughput
- 25th-percentile latency
- 50th-percentile latency
- 75th-percentile latency

And we do also support a custom percentile, you can setup that by passing a `percentile` in the `BaseModelInspector` constructor. 