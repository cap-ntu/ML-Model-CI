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

from modelci.types.bo import Framework, Engine, Status
from modelci.types.trtis_objects import ModelInputFormat


class IOShapeVO(BaseModel):
    shape: List[int]
    dtype: str
    name: str = None
    format: ModelInputFormat = ModelInputFormat.FORMAT_NONE


class InfoTupleVO(BaseModel):
    avg: float
    p50: float
    p95: float
    p99: float


class ProfileMemoryVO(BaseModel):
    total_memory: int
    memory_usage: int
    utilization: float


class ProfileLatencyVO(BaseModel):
    init_latency: InfoTupleVO
    preprocess_latency: InfoTupleVO
    inference_latency: InfoTupleVO
    postprocess_latency: InfoTupleVO


class ProfileThroughputVO(BaseModel):
    batch_formation_throughput: InfoTupleVO
    preprocess_throughput: InfoTupleVO
    inference_throughput: InfoTupleVO
    postprocess_throughput: InfoTupleVO


class DynamicResultVO(BaseModel):
    device_id: str
    device_name: str
    batch: int
    memory: ProfileMemoryVO
    latency: ProfileLatencyVO
    throughput: ProfileThroughputVO
    ip: str
    create_time: datetime


class ProfileResultVO(BaseModel):
    static_result: str
    dynamic_results: List[DynamicResultVO]


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
    profile_result: ProfileResultVO
    status: Status
    create_time: datetime


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
