from datetime import datetime
from typing import List, Optional

from pydantic import Field
from pydantic.main import BaseModel
from pydantic.types import PositiveInt, NonNegativeFloat, confloat, NonNegativeInt, conint


class InfoTuple(BaseModel):
    """A triplet tuple containing overall, average, 50th percentile, 95th percentile, and 99th percentile values of a
    data over a period of time.
    """
    # the average value
    avg: float
    # 50th percentile, 95th percentile, and 99th percentile values,  of a data.
    p50: float
    p95: float
    p99: float

    def __init__(self, avg: float, p50: float, p95: float, p99: float):
        super().__init__(avg=avg, p50=p50, p95=p95, p99=p99)


class ProfileMemory(BaseModel):
    """Memory class in dynamic profiling result
    """
    # Main or GPU memory consumption in Byte for loading and initializing the model
    total_memory: NonNegativeInt
    # GPU memory consumption in Byte for processing batch data
    memory_usage: NonNegativeInt
    # GPU utilization rate for processing batch data
    utilization = confloat(ge=0, le=1)


class ProfileLatency(BaseModel):
    """End-to-end latency in dynamic profiling result
    """
    # Min, max and avg model loading and initialization latencies
    initialization_latency: InfoTuple
    # Min, max and avg preprocess latencies
    preprocess_latency: InfoTuple
    # Min, max and avg inference latencies
    inference_latency: InfoTuple
    # Min, max and avg postprocess latencies
    postprocess_latency: InfoTuple


class ProfileThroughput(BaseModel):
    """End-to-end throughput in dynamic profiling result
    """
    # Batch formation QPS
    batch_formation_throughput: NonNegativeFloat= Field(default=0)
    # Batch preprocess QPS
    preprocess_throughput: NonNegativeFloat= Field(default=0)
    # Batch inference QPS
    inference_throughput: NonNegativeFloat= Field(default=0)
    # Batch postprocess QPS
    postprocess_throughput: NonNegativeFloat= Field(default=0)


class DynamicProfileResult(BaseModel):
    """
    Dynamic profiling result

    The primary key of the document is (ip, device_id) pair.
    """
    # IP address of the cluster node
    ip: str=Field(default='127.0.0.1')
    # Device ID, e.g. cpu, cuda:0, cuda:1
    device_id: str
    # Device name, e.g. Tesla K40c
    device_name: str
    # Batch size
    batch: conint(ge=1)
    # Memory class in dynamic profiling result
    memory: ProfileMemory
    # End-to-end latency in dynamic profiling result
    latency: ProfileLatency
    # End-to-end throughput in dynamic profiling result
    throughput: ProfileThroughput
    # Creation time of this record
    create_time: Optional[datetime] = Field(default_factory=datetime.utcnow)


class StaticProfileResult(BaseModel):
    """
    Static profiling result
    """

    # Number of parameters of this model
    parameters: PositiveInt
    # Floating point operations
    flops: PositiveInt
    # Memory consumption in Byte in order to load this model into GPU or CPU
    memory: PositiveInt
    # Memory read in Byte
    mread: PositiveInt
    # Memory write in Byte
    mwrite: PositiveInt
    # Memory readwrite in Byte
    mrw: PositiveInt


class ProfileResult(BaseModel):
    """
    Profiling result
    """
    # Static profile result
    static_profile_result: Optional[StaticProfileResult]
    # Dynamic profile result
    dynamic_profile_results: Optional[List[DynamicProfileResult]] = Field(default_factory=list)
