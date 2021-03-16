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
from typing import Union, Optional, Dict, List, Any

from bson import ObjectId
from gridfs import GridOut
from pydantic import BaseModel, FilePath, DirectoryPath, PositiveInt, Field, root_validator

from .common import Metric, IOShape, Framework, Engine, Task, ModelStatus, Status, PydanticObjectId, \
    named_enum_json_encoder
from .pattern import as_form
from ...hub.utils import parse_path_plain, generate_path_plain


class Weight(BaseModel):
    """TODO: Only works for MLModelIn"""

    __slots__ = ('file',)

    __root__: Optional[PydanticObjectId]

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
        return ''

    def __bytes__(self):
        if self.file:
            return self.file.read_bytes()


@as_form
class BaseMLModel(BaseModel):
    architecture: str = Field(..., example='ResNet50')
    framework: Framework
    engine: Engine
    version: PositiveInt = Field(..., example=1)
    dataset: str = Field(..., example='ImageNet')
    metric: Dict[Metric, float] = Field(..., example='{"acc": 0.76}')
    task: Task
    inputs: List[IOShape] = Field(
        ...,
        example='[{"name": "input", "shape": [-1, 3, 224, 224], "dtype": "TYPE_FP32", "format": "FORMAT_NCHW"}]'
    )
    outputs: List[IOShape] = Field(
        ...,
        example='[{"name": "output", "shape": [-1, 1000], "dtype": "TYPE_FP32"}]'
    )

    def dict(self, use_enum_values: bool = False, **kwargs):
        """
        Args:
            use_enum_values: Export the model as dict with included :class:`enum.Enum` to be their value.
            **kwargs: Other keyword arguments in :meth:`pydantic.BaseModel.dict`.
        """
        if use_enum_values:
            self.Config.use_enum_values = True
            # TODO: auto find field who contains `Enum`
            IOShape.Config.use_enum_values = True
            data = super().dict(**kwargs)
            self.Config.use_enum_values = False
            IOShape.Config.use_enum_values = False
        else:
            data = super().dict(**kwargs)

        # fix metric key as a Enum
        metric: dict = data.get('metric', None)
        if metric:
            data['metric'] = {Metric(k).name: v for k, v in metric.items() }

        return data

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str,
            Framework: named_enum_json_encoder,
            Engine: named_enum_json_encoder,
            Task: named_enum_json_encoder,
            Status: named_enum_json_encoder,
            ModelStatus: named_enum_json_encoder,
            # TODO: check whether we can auto detect sub-model's json encoder
            **IOShape.__config__.json_encoders,
        }

    @property
    def saved_path(self):
        return generate_path_plain(self.architecture, self.task, self.framework, self.engine, self.version)


class MLModel(BaseMLModel):
    id: Optional[PydanticObjectId] = Field(default=None, alias='_id')
    parent_model_id: Optional[PydanticObjectId]
    weight: Weight
    profile_result: Optional[Any]
    status: Optional[Status] = Status.Unknown
    model_input: Optional[list]  # TODO: merge into field `inputs`
    model_status: Optional[List[ModelStatus]] = Field(default_factory=list)
    creator: Optional[str] = Field(default_factory=getpass.getuser)
    create_time: Optional[datetime] = Field(default_factory=datetime.utcnow)

    @property
    def saved_path(self):
        suffix = Path(self.weight.filename).suffix
        return super().saved_path.with_suffix(suffix)


class MLModelFromYaml(BaseMLModel):
    weight: Union[FilePath, DirectoryPath]
    architecture: Optional[str]
    framework: Optional[Framework]
    engine: Optional[Engine]
    task: Optional[Task]
    version: Optional[int]
    convert: Optional[bool] = True
    profile: Optional[bool] = False

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

    @property
    def saved_path(self):
        suffix = Path(self.weight).suffix
        return super().saved_path.with_suffix(suffix)
