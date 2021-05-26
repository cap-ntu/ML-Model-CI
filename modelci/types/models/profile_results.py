import ipaddress
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from modelci.types.models.common import InfoTuple


class ProfileMemory(BaseModel):
    """Memory class in dynamic profiling result business object.

    Args:
        total_memory (int): Total memory in Byte.
        memory_usage (int): GPU memory used in Byte.
        utilization (float): GPU utilization.
    """

    total_memory: int
    memory_usage: int
    utilization: float


class ProfileLatency(BaseModel):
    """End-to-end latency in dynamic profiling result business object.

    This class records the end-to-end latency in four stages: data engine initialization, pre-processing, inference, and
    post-processing.

    TODO: support Iterable args, refer to dynamic_profile_result_bo.py

    Args:
        init_latency (InfoTuple): initialization latency.
        preprocess_latency (InfoTuple): pre-processing latency.
        inference_latency (InfoTuple): inference latency.
        postprocess_latency (InfoTuple): post-processing latency.
    """
    init_latency: InfoTuple = InfoTuple(avg=0, p50=0, p95=0, p99=0)
    preprocess_latency: InfoTuple = InfoTuple(avg=0, p50=0, p95=0, p99=0)
    inference_latency: InfoTuple = InfoTuple(avg=0, p50=0, p95=0, p99=0)
    postprocess_latency: InfoTuple = InfoTuple(avg=0, p50=0, p95=0, p99=0)


class ProfileThroughput(BaseModel):
    """End-to-end throughput in dynamic profiling result business object.

    This class records the end-to-end throughput in four categories: data to batched data, pre-processing, inference,
    and post-processing.

    TODO: re-organize

    Args:
        batch_formation_throughput (Union[InfoTuple, Iterable]): data to batched data throughput.
        preprocess_throughput (Union[InfoTuple, Iterable]): pre-processing throughput.
        inference_throughput (Union[InfoTuple, Iterable]): inference throughput.
        postprocess_throughput (Union[InfoTuple, Iterable]): post-processing throughput.
    """
    batch_formation_throughput: float = 0
    preprocess_throughput: float = 0
    inference_throughput: float = 0
    postprocess_throughput: float = 0


class DynamicProfileResult(BaseModel):
    """Dynamic profiling result business object.

    Args:
        device_id (str): Device ID. e.g. cuda:0.
        device_name (str): Device name. e.g. Tesla K40c.
        batch (int): Batch size.
        memory (ProfileMemory): Memory.
        latency (ProfileLatency): Latency.
        throughput (ProfileThroughput): Throughput.
        ip (Optional[str]): IP address. Default to 'localhost' (i.e. 127.0.0.1).
        create_time (Optional[datetime]): Create time. Default to current datetime.
    """
    device_id: str
    device_name: str
    batch: int
    memory: ProfileMemory
    latency: ProfileLatency
    throughput: ProfileThroughput
    ip: str = '127.0.0.1'
    create_time: datetime = datetime.now()

    def __init__(self, **data: Any):
        """Initializer."""
        # IP address
        super().__init__(**data)
        self.ip = ipaddress.ip_address(self.ip)
