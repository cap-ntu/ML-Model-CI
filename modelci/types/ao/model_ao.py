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


class IOShapeAO(BaseModel):
    shape: List[int]
    dtype: str
    name: str = None
    format: ModelInputFormat = ModelInputFormat.FORMAT_NONE


class ProfileResultAO(BaseModel):
    pass


class ModelOut(BaseModel):
    id: str
    name: str
    framework: Framework
    engine: Engine
    version: int
    dataset: str
    acc: float
    task: str
    inputs: List[IOShapeAO]
    outputs: List[IOShapeAO]
    profile_result: ProfileResultAO
    status: Status
    create_time: datetime


class ModelIn(BaseModel):
    dataset: str
    acc: float
    task: str
    inputs: List[IOShapeAO]
    outputs: List[IOShapeAO]
    architecture: str
    framework: Framework
    version: int
    convert: bool
    profile: bool
