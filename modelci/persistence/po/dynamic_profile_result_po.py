from mongoengine import EmbeddedDocument
from mongoengine.fields import *


class DynamicProfileResultPO(EmbeddedDocument):
    """
    Dynamic profiling result plain object.

    The primary key of the document is (ip, device_id) pair.
    """
    # IP address of the cluster node
    ip = StringField(required=True)
    # Device ID, e.g. cpu, cuda:0, cuda:1
    device_id = StringField(required=True)
    # Device name, e.g. Tesla K40c
    device_name = StringField(required=True)
    # Batch size
    batch = IntField(required=True)
    # Main or GPU memory consumption in MB for loading and initializing the model
    memory = FloatField(required=True)
    # Main memory consumption in MB for processing batch data
    cpu_memory = FloatField(required=True)
    # GPU memory consumption in MB for processing batch data
    gpu_memory = FloatField(required=True)
    # Min, max and avg model loading and initialization latencies
    initialization_latency = ListField(FloatField(), required=True)
    # Min, max and avg preprocess latencies
    preprocess_latency = ListField(FloatField(), required=True)
    # Min, max and avg inference latencies
    inference_latency = ListField(FloatField(), required=True)
    # Min, max and avg postprocess latencies
    postprocess_latency = ListField(FloatField(), required=True)
    # Min, max and avg batch formation latencies
    batch_formation_throughput = ListField(FloatField(), required=True)
    # Min, max and avg batch preprocess latencies
    preprocess_throughput = ListField(FloatField(), required=True)
    # Min, max and avg batch inference latencies
    inference_throughput = ListField(FloatField(), required=True)
    # Min, max and avg batch postprocess latencies
    postprocess_throughput = ListField(FloatField(), required=True)
    # Creation time of this record
    create_time = DateTimeField(required=True)

    meta = {
        'indexes': [
            {'fields': ('ip', 'device_id'), 'unique': True}
        ],
    }
