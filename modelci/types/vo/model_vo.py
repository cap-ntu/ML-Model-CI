#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/19/2020
"""
from datetime import datetime
from typing import List

from pydantic import BaseModel

from modelci.types.bo import Framework, Engine, Status, ModelBO, IOShape, ProfileResultBO, DynamicProfileResultBO, \
    InfoTuple, ProfileMemory, ProfileLatency, ProfileThroughput
from modelci.types.trtis_objects import ModelInputFormat


class IOShapeVO(BaseModel):
    shape: List[int]
    dtype: str
    name: str = None
    format: ModelInputFormat = ModelInputFormat.FORMAT_NONE

    @staticmethod
    def from_bo(io_shape: IOShape):
        return IOShapeVO(**vars(io_shape))


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
        return ProfileLatencyVO(**vars(profile_memory_bo))


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
    batch_formation_throughput: InfoTupleVO
    preprocess_throughput: InfoTupleVO
    inference_throughput: InfoTupleVO
    postprocess_throughput: InfoTupleVO

    @staticmethod
    def from_bo(profile_throughput: ProfileThroughput):
        return ProfileThroughputVO(
            batch_formation_throughput=InfoTupleVO.from_bo(profile_throughput.batch_formation_throughput),
            preprocess_throughput=InfoTupleVO.from_bo(profile_throughput.preprocess_throughput),
            inference_throughput=InfoTupleVO.from_bo(profile_throughput.inference_throughput),
            postprocess_throughput=InfoTupleVO.from_bo(profile_throughput.postprocess_throughput),
        )


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
            memeory=ProfileMemoryVO.from_bo(dynamic_profile_result_bo.memory),
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

        return ProfileLatencyVO(
            static_result='N.A.',
            dynamic_restuls=list(map(DynamicResultVO.from_bo, profile_result_bo.dynamic_results))
        )


class ModelListOut(BaseModel):
    id: str
    name: str
    framework: Framework
    engine: Engine
    version: int
    dataset: str
    acc: float
    task: str
    inputs: List[IOShapeVO]
    outputs: List[IOShapeVO]
    status: Status
    create_time: datetime

    @staticmethod
    def from_bo(model_bo: ModelBO):
        return ModelListOut(
            id=model_bo.id,
            name=model_bo.name,
            framework=model_bo.framework,
            engine=model_bo.engine,
            version=str(model_bo.version),
            dataset=model_bo.dataset,
            acc=model_bo.acc,
            task=model_bo.task,
            inputs=list(map(IOShapeVO.from_bo, model_bo.inputs)),
            outputs=list(map(IOShapeVO.from_bo, model_bo.outputs)),
            status=model_bo.status,
            create_time=model_bo.create_time,
        )


class ModelDetailOut(BaseModel):
    id: str
    name: str
    framework: Framework
    engine: Engine
    version: int
    dataset: str
    acc: float
    task: str
    inputs: List[IOShapeVO]
    outputs: List[IOShapeVO]
    profile_result: ProfileResultVO = None
    status: Status
    create_time: datetime

    @staticmethod
    def from_bo(model_bo: ModelBO):
        return ModelDetailOut(
            id=model_bo.id,
            name=model_bo.name,
            framework=model_bo.framework,
            engine=model_bo.engine,
            version=str(model_bo.version),
            dataset=model_bo.dataset,
            acc=model_bo.acc,
            task=model_bo.task,
            inputs=list(map(IOShapeVO.from_bo, model_bo.inputs)),
            outputs=list(map(IOShapeVO.from_bo, model_bo.outputs)),
            profile_result=ProfileResultVO.from_bo(model_bo.profile_result),
            status=model_bo.status,
            create_time=model_bo.create_time,
        )


class ModelIn(BaseModel):
    dataset: str
    acc: float
    task: str
    inputs: List[IOShapeVO]
    outputs: List[IOShapeVO]
    architecture: str
    framework: Framework
    version: int
    convert: bool
    profile: bool
