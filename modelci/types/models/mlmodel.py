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

from modelci.hub.utils import parse_path_plain, generate_path
from .common import Metric, IOShape, Framework, Engine, Task, ModelStatus, Status


class Weight(BaseModel):
    """TODO: Only works for MLModelIn"""

    id: Optional[str]
    file: Optional[FilePath]
    _grid_out: Optional[GridOut]

    @classmethod
    def validate(cls, value):
        data = dict()
        if isinstance(value, str):
            data['id'] = value
        elif isinstance(value, Path):
            data['file'] = value
        else:
            data = value
        return super().validate(data)

    @root_validator(pre=True)
    def check_id_or_bytes(cls, values):  # pylint: disable=no-self-use
        id_, file = values.get('id', None), values.get('file', None)
        if not id_ and file:
            return values
        elif id_ and not file:
            return values
        else:
            raise ValueError('Not found either field `id` nor field `file`. '
                             'You should set one of [`id`, `file`]')

    @property
    def filename(self):
        if self.file:
            return self.file.name

    def __bytes__(self):
        if self.file:
            return self.file.read_bytes()

    def dict(self, **kwargs):
        attrs = super().dict(**kwargs)
        return attrs.pop('id', None)


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

    class Config:
        json_encoders = {
            Framework: named_enum_json_encoder,
            Engine: named_enum_json_encoder,
            Metric: named_enum_json_encoder,
            Status: named_enum_json_encoder,
        }


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

    class Config:
        json_encoders = {
            Framework: named_enum_json_encoder,
            Engine: named_enum_json_encoder,
            Metric: named_enum_json_encoder,
            Task: named_enum_json_encoder,
            Status: named_enum_json_encoder,
            Weight: None,
        }

    @property
    def saved_path(self):
        return generate_path(self.architecture, self.task, self.framework, self.engine, self.version)


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
