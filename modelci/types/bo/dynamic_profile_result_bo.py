# -*- coding: utf-8 -*-
"""Dynamic profiling result business object.

This module defines the classes used for dynamic profiling result in business layer.
"""
import ipaddress
from datetime import datetime
from typing import Union, Iterable

from modelci.types.do import DynamicProfileResultDO
from .model_objects import InfoTuple


class ProfileMemory(object):
    """Memory class in dynamic profiling result business object.
    """

    def __init__(self, memory: int, cpu_memory: int, gpu_memory: int):
        """Initializer.

        Args:
            memory (int): Total memory in Byte.
            cpu_memory (int): CPU memory in Byte.
            gpu_memory (int): GPU memory in Byte.
        """
        # total memory
        self.memory = memory
        # CPU memory
        self.cpu_memory = cpu_memory
        # GPU memory
        self.gpu_memory = gpu_memory


class ProfileLatency(object):
    """End-to-end latency in dynamic profiling result business object.

    This class records the end-to-end latency in four stages: data engine initialization, pre-processing, inference, and
    post-processing.
    """

    def __init__(self,
                 init_latency: Union[InfoTuple, Iterable],
                 preprocess_latency: Union[InfoTuple, Iterable],
                 inference_latency: Union[InfoTuple, Iterable],
                 postprocess_latency: Union[InfoTuple, Iterable]):
        """
        Initializer.

        Args:
            init_latency (Union[InfoTuple, Iterable]): initialization latency.
                An `InfoTuple` instance or a iterable instance containing min, max and average throughput.
            preprocess_latency (Union[InfoTuple, Iterable]): pre-processing latency.
                Requirements are the same as `init_latency`.
            inference_latency (Union[InfoTuple, Iterable]): inference latency.
                Requirements are the same as `init_latency`.
            postprocess_latency (Union[InfoTuple, Iterable]): post-processing latency.
                Requirements are the same as `init_latency`.
        """
        # convert latencies to InfoTuple type
        if isinstance(init_latency, Iterable):
            init_latency = InfoTuple(tuple(init_latency))
        if isinstance(preprocess_latency, Iterable):
            preprocess_latency = InfoTuple(tuple(preprocess_latency))
        if isinstance(inference_latency, Iterable):
            inference_latency = InfoTuple(tuple(inference_latency))
        if isinstance(postprocess_latency, Iterable):
            postprocess_latency = InfoTuple(tuple(postprocess_latency))

        # initialization latency
        self.init_latency = init_latency
        # pre-processing latency
        self.preprocess_latency = preprocess_latency
        # inference latency
        self.inference_latency = inference_latency
        # post-precessing latency
        self.postprocess_latency = postprocess_latency


class ProfileThroughput(object):
    """End-to-end throughput in dynamic profiling result business object.

    This class records the end-to-end throughput in four categories: data to batched data, pre-processing, inference,
    and post-processing.
    """

    def __init__(self,
                 batch_formation_throughput: Union[InfoTuple, Iterable],
                 preprocess_throughput: Union[InfoTuple, Iterable],
                 inference_throughput: Union[InfoTuple, Iterable],
                 postprocess_throughput: Union[InfoTuple, Iterable]
                 ):
        """
        Initializer.

        Args:
            batch_formation_throughput (Union[InfoTuple, Iterable]): data to batched data throughput.
                An `InfoTuple` instance or a iterable instance containing min, max and average throughput.
            preprocess_throughput (Union[InfoTuple, Iterable]): pre-processing throughput.
                Requirements are the same as `batch_formation_throughput`.
            inference_throughput (Union[InfoTuple, Iterable]): inference throughput.
                Requirements are the same as `batch_formation_throughput`.
            postprocess_throughput (Union[InfoTuple, Iterable]): post-processing throughput.
                Requirements are the same as `batch_formation_throughput`.
        """
        # convert latencies to InfoTuple type
        if isinstance(batch_formation_throughput, Iterable):
            batch_formation_throughput = InfoTuple(tuple(batch_formation_throughput))
        if isinstance(preprocess_throughput, Iterable):
            preprocess_throughput = InfoTuple(tuple(preprocess_throughput))
        if isinstance(inference_throughput, Iterable):
            inference_throughput = InfoTuple(tuple(inference_throughput))
        if isinstance(postprocess_throughput, Iterable):
            postprocess_throughput = InfoTuple(tuple(postprocess_throughput))

        # data to batched data throughput
        self.batch_formation_throughput = batch_formation_throughput
        # inference throughput
        self.inference_throughput = inference_throughput
        # pre-processing throughput
        self.preprocess_throughput = preprocess_throughput
        # post-processing throughput
        self.postprocess_throughput = postprocess_throughput


class DynamicProfileResultBO(object):
    """Dynamic profiling result business object.
    """

    def __init__(self,
                 device_id: str,
                 device_name: str,
                 batch: int,
                 memory: ProfileMemory,
                 latency: ProfileLatency,
                 throughput: ProfileThroughput,
                 ip: str = '127.0.0.1',
                 create_time: datetime = datetime.now()):
        """
        Initializer.

        Args:
            device_id (str): Device ID. e.g. cuda:0.
            device_name (str): Device name. e.g. Tesla K40c.
            batch (int): Batch size.
            memory (ProfileMemory): Memory.
            throughput (ProfileThroughput): Throughput.
            ip (Optional[str]): IP address. Default to 'localhost' (i.e. 127.0.0.1).
            create_time (Optional[datetime]): Create time. Default to current datetime.
        """
        # IP address
        self.ip = ipaddress.ip_address(ip)
        self.device_id = device_id
        self.device_name = device_name
        self.batch = batch
        self.memory = memory
        self.latency = latency
        self.throughput = throughput
        self.create_time = create_time

    def to_dynamic_profile_result_po(self):
        """Convert business object to plain object for persistence.
        """
        dpr_po = DynamicProfileResultDO(
            ip=str(self.ip),
            device_id=self.device_id,
            device_name=self.device_name,
            batch=self.batch,
            memory=self.memory.memory,
            cpu_memory=self.memory.cpu_memory,
            gpu_memory=self.memory.gpu_memory,
            initialization_latency=self.latency.init_latency.tolist(),
            preprocess_latency=self.latency.preprocess_latency.tolist(),
            inference_latency=self.latency.inference_latency.tolist(),
            postprocess_latency=self.latency.postprocess_latency.tolist(),
            batch_formation_throughput=self.throughput.batch_formation_throughput.tolist(),
            preprocess_throughput=self.throughput.preprocess_throughput.tolist(),
            inference_throughput=self.throughput.inference_throughput.tolist(),
            postprocess_throughput=self.throughput.postprocess_throughput.tolist(),
            create_time=self.create_time
        )

        return dpr_po

    @staticmethod
    def from_dynamic_profile_result_po(dpr_po: DynamicProfileResultDO):
        """Create a business object from plain object.

        Args:
            dpr_po (DynamicProfileResultPO): dynamic profiling result plain object to be converted.

        Return:
            Dynamic profiling result business object. None if None is input.
        """
        # null safe
        if dpr_po is None:
            return None

        dpr = DynamicProfileResultBO(
            ip=dpr_po.ip,
            device_id=dpr_po.device_id,
            device_name=dpr_po.device_name,
            batch=dpr_po.batch,
            memory=ProfileMemory(
                memory=dpr_po.memory,
                cpu_memory=dpr_po.cpu_memory,
                gpu_memory=dpr_po.gpu_memory
            ),
            latency=ProfileLatency(
                init_latency=dpr_po.initialization_latency,
                preprocess_latency=dpr_po.preprocess_latency,
                inference_latency=dpr_po.inference_latency,
                postprocess_latency=dpr_po.postprocess_latency
            ),
            throughput=ProfileThroughput(
                batch_formation_throughput=dpr_po.batch_formation_throughput,
                preprocess_throughput=dpr_po.preprocess_throughput,
                inference_throughput=dpr_po.inference_throughput,
                postprocess_throughput=dpr_po.postprocess_throughput
            ))
        return dpr
