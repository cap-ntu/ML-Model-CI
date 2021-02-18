#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/19/2020
"""
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Union

from pydantic import BaseModel

from modelci.types.bo import (
    ModelBO,
    IOShape,
    ProfileResultBO,
    DynamicProfileResultBO,
    InfoTuple,
    ProfileMemory,
    ProfileLatency,
    ProfileThroughput,
)


class CaseInsensitiveEnum(Enum):
    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member


class ModelInputFormat(CaseInsensitiveEnum):
    FORMAT_NONE = 'FORMAT_NONE'
    FORMAT_NHWC = 'FORMAT_NHWC'
    FORMAT_NCHW = 'FORMAT_NCHW'


class Task(CaseInsensitiveEnum):
    IMAGE_CLASSIFICATION = 'Image Classification'
    OBJECT_DETECTION = 'Object Detection'
    SEGMENTATION = 'Segmentation'


class Metric(CaseInsensitiveEnum):
    ACC = 'acc'
    MAP = 'mAp'
    IOU = 'IoU'


class Framework(CaseInsensitiveEnum):
    TENSORFLOW = 'TensorFlow'
    PYTORCH = 'PyTorch'


class Engine(CaseInsensitiveEnum):
    NONE = 'None'
    TFS = 'TFS'
    TORCHSCRIPT = 'TorchScript'
    ONNX = 'ONNX'
    TRT = 'TRT'
    TVM = 'TVM'
    CUSTOMIZED = 'CUSTOMIZED'
    PYTORCH = 'PYTORCH'


class Status(CaseInsensitiveEnum):
    UNKNOWN = 'Unknown'
    PASS = 'Pass'
    RUNNING = 'Running'
    FAIL = 'Fail'


class ModelStatus(CaseInsensitiveEnum):
    PUBLISHED = 'Published'
    CONVERTED = 'Converted'
    PROFILING = 'Profiling'
    IN_SERVICE = 'In Service'
    DRAFT = 'Draft'
    VALIDATING = 'Validating'
    TRAINING = 'Training'


class IOShapeVO(BaseModel):
    shape: List[int]
    dtype: str
    name: str = None
    format: ModelInputFormat = ModelInputFormat.FORMAT_NONE

    @staticmethod
    def from_bo(io_shape: IOShape):
        io_shape_data = vars(io_shape)
        io_shape_data['format'] = ModelInputFormat(io_shape.format.name)
        return IOShapeVO(**io_shape_data)


class InfoTupleVO(BaseModel):
    avg: float
    p50: float
    p95: float
    p99: float

    @staticmethod
    def from_bo(info_tuple: InfoTuple):
        return InfoTupleVO(**vars(info_tuple))


class ProfileMemoryVO(BaseModel):
    total_memory: int
    memory_usage: int
    utilization: float

    @staticmethod
    def from_bo(profile_memory_bo: ProfileMemory):
        return ProfileMemoryVO(**vars(profile_memory_bo))


class ProfileLatencyVO(BaseModel):
    init_latency: InfoTupleVO
    preprocess_latency: InfoTupleVO
    inference_latency: InfoTupleVO
    postprocess_latency: InfoTupleVO

    @staticmethod
    def from_bo(profile_latency: ProfileLatency):
        return ProfileLatencyVO(
            init_latency=InfoTupleVO.from_bo(profile_latency.init_latency),
            preprocess_latency=InfoTupleVO.from_bo(profile_latency.init_latency),
            inference_latency=InfoTupleVO.from_bo(profile_latency.inference_latency),
            postprocess_latency=InfoTupleVO.from_bo(profile_latency.postprocess_latency),
        )


class ProfileThroughputVO(BaseModel):
    batch_formation_throughput: float
    preprocess_throughput: float
    inference_throughput: float
    postprocess_throughput: float

    @staticmethod
    def from_bo(profile_throughput: ProfileThroughput):
        return ProfileThroughputVO(**vars(profile_throughput))


class DynamicResultVO(BaseModel):
    device_id: str
    device_name: str
    batch: int
    memory: ProfileMemoryVO
    latency: ProfileLatencyVO
    throughput: ProfileThroughputVO
    ip: str
    create_time: datetime

    @staticmethod
    def from_bo(dynamic_profile_result_bo: DynamicProfileResultBO):
        return DynamicResultVO(
            device_id=dynamic_profile_result_bo.device_id,
            device_name=dynamic_profile_result_bo.device_name,
            batch=dynamic_profile_result_bo.batch,
            memory=ProfileMemoryVO.from_bo(dynamic_profile_result_bo.memory),
            latency=ProfileLatencyVO.from_bo(dynamic_profile_result_bo.latency),
            throughput=ProfileThroughputVO.from_bo(dynamic_profile_result_bo.throughput),
            ip=str(dynamic_profile_result_bo.ip),
            create_time=dynamic_profile_result_bo.create_time,
        )


class ProfileResultVO(BaseModel):
    # TODO: to be added
    static_result: str
    dynamic_results: List[DynamicResultVO]

    @staticmethod
    def from_bo(profile_result_bo: ProfileResultBO):
        if profile_result_bo is None:
            return None

        return ProfileResultVO(
            static_result='N.A.',
            dynamic_results=list(map(DynamicResultVO.from_bo, profile_result_bo.dynamic_results))
        )


class ModelListOut(BaseModel):
    id: str
    name: str
    framework: Framework
    engine: Engine
    version: Union[int, str]
    dataset: str
    metric: Dict[Metric, float]
    task: Task
    inputs: List[IOShapeVO]
    outputs: List[IOShapeVO]
    parent_model_id: Optional[str] = None
    profile_result: ProfileResultVO = None
    status: Status
    model_status: List[ModelStatus]
    creator: str
    create_time: datetime

    @staticmethod
    def from_bo(model_bo: ModelBO):
        return ModelListOut(
            id=model_bo.id,
            name=model_bo.name,
            framework=Framework(model_bo.framework.name),
            engine=Engine(model_bo.engine.name),
            version=str(model_bo.version),
            dataset=model_bo.dataset,
            metric={Metric(key.name): val for key, val in model_bo.metric.items()},
            task=Task(model_bo.task.name),
            parent_model_id=model_bo.parent_model_id,
            inputs=list(map(IOShapeVO.from_bo, model_bo.inputs)),
            outputs=list(map(IOShapeVO.from_bo, model_bo.outputs)),
            profile_result=ProfileResultVO.from_bo(model_bo.profile_result),
            status=Status(model_bo.status.name),
            model_status=[ModelStatus(item.name) for item in model_bo.model_status],
            creator=model_bo.creator,
            create_time=model_bo.create_time,
        )


class ModelDetailOut(BaseModel):
    id: str
    name: str
    framework: Framework
    engine: Engine
    version: Union[int, str]
    dataset: str
    metric: Dict[Metric, float]
    task: Task
    parent_model_id: str
    inputs: List[IOShapeVO]
    outputs: List[IOShapeVO]
    profile_result: ProfileResultVO = None
    status: Status
    model_status: List[ModelStatus]
    creator: str
    create_time: datetime

    @staticmethod
    def from_bo(model_bo: ModelBO):
        return ModelDetailOut(
            id=model_bo.id,
            name=model_bo.name,
            framework=Framework(model_bo.framework.name),
            engine=Engine(model_bo.engine.name),
            version=str(model_bo.version),
            dataset=model_bo.dataset,
            metric={Metric(key.name): val for key, val in model_bo.metric.items()},
            task=Task(model_bo.task.name),
            parent_model_id=model_bo.parent_model_id,
            inputs=list(map(IOShapeVO.from_bo, model_bo.inputs)),
            outputs=list(map(IOShapeVO.from_bo, model_bo.outputs)),
            profile_result=ProfileResultVO.from_bo(model_bo.profile_result),
            status=Status(model_bo.status.name),
            model_status=[ModelStatus(item) for item in model_bo.model_status],
            creator=model_bo.creator,
            create_time=model_bo.create_time,
        )
