#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/17/2021
"""
import getpass
import os
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Dict, List

from gridfs import GridOut
from pydantic import BaseModel, FilePath, DirectoryPath, Field, root_validator

from modelci.hub.utils import parse_path_plain, generate_path_plain
from .common import Metric, IOShape, Framework, Engine, Task, ModelStatus, Status


class Weight(BaseModel):
    """TODO: Only works for MLModelIn"""

    __slots__ = ('file',)

    __root__: Optional[str]

    def __init__(self, __root__):
        if isinstance(__root__, Path):
            object.__setattr__(self, 'file', FilePath.validate(__root__))
            __root__ = None

        self._grid_out: Optional[GridOut]

        super().__init__(__root__=__root__)

    @property
    def filename(self):
        if self.file:
            return self.file.name

    def __bytes__(self):
        if self.file:
            return self.file.read_bytes()


def named_enum_json_encoder(v):
    return v.name()


class MLModel(BaseModel):
    id: Optional[str]
    architecture: str
    framework: Framework
    engine: Engine
    version: int
    dataset: str
    metric: Dict[Metric, float]
    task: Task
    inputs: List[IOShape]
    outputs: List[IOShape]
    weight: Weight
    # profile_result: Optional[ProfileResult]
    status: Status = Status.UNKNOWN
    model_status: List[ModelStatus] = Field(default_factory=list)
    creator: str = Field(default_factory=getpass.getuser)
    create_time: datetime = Field(default_factory=datetime.now, const=True)

    def dict(self, **kwargs):
        MLModelIn.Config.use_enum_values = True
        data = super().dict(**kwargs)
        # fix metric key as a Enum
        metric: dict = data.get('metric', None)
        if metric:
            data['metric'] = {k.name: v for k, v in metric.items()}
        MLModelIn.Config.use_enum_values = False
        return data


class MLModelIn(BaseModel):
    # noinspection PyUnresolvedReferences
    """
    Attributes:
        parent_model_id: The parent model ID of current model if this model is derived from a pre-existing one.
    """
    weight: Weight
    dataset: str
    metric: Dict[Metric, float]
    parent_model_id: Optional[str]
    inputs: List[IOShape]
    outputs: List[IOShape]
    model_input: Optional[list]  # TODO: merge into field `inputs`
    architecture: str
    framework: Framework
    engine: Engine
    task: Task
    version: int
    model_status: List[ModelStatus] = Field(default_factory=list)

    @property
    def saved_path(self):
        return generate_path_plain(self.architecture, self.task, self.framework, self.engine, self.version)

    def dict(self, **kwargs):
        MLModelIn.Config.use_enum_values = True
        data = super().dict(**kwargs)
        # fix metric key as a Enum
        metric: dict = data.get('metric', None)
        if metric:
            data['metric'] = {k.name: v for k, v in metric.items()}
        MLModelIn.Config.use_enum_values = False
        return data


class MLModelInYaml(MLModelIn):
    weight: Union[FilePath, DirectoryPath]
    architecture: Optional[str]
    framework: Optional[Framework]
    engine: Optional[Engine]
    task: Optional[Task]
    version: Optional[int]

    @root_validator(pre=True)
    def check_model_info(cls, values: dict):  # pylint: disable=no-self-use
        """
        Check provided model info is consistent with the one inferred from weight.

        This validator also auto fill-in implicit model info from weight.
        """
        weight = values.get('weight', None)
        if weight:
            weight = os.path.expanduser(weight)
            values['weight'] = weight

        model_info_provided = {
            k: values.get(k, None) for k in ('architecture', 'framework', 'engine', 'task', 'version')
        }

        # fill in implicit model info from weight path
        if not all(model_info_provided.values()):
            model_info = parse_path_plain(weight)
            for k, v in model_info_provided.items():
                if not v:
                    values[k] = model_info[k]
                elif v != model_info[k]:
                    raise ValueError(f'{k} expected to be {model_info[k]} inferred from {weight}, but got {v}.')

        return values
