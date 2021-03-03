#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/26/2021
"""

import warnings
from enum import Enum, unique
from typing import List

import numpy as np
import tensorflow as tf
import torch
from bson import ObjectId
from pydantic import BaseModel

from modelci.types.trtis_objects import DataType


class NamedEnum(Enum):
    """
    A enumerator that can be initialized by name.

    Examples:
        Create a Color Enum:
        >>> class Color(NamedEnum):
        ...     RED = 0
        ...     GREEN = 1
        ...     BLUE = 2

        You can create a Color enum object by its name:
        >>> color = Color('RED')
        ... print(color)
        <Color.RED: 0>
    """

    @classmethod
    def get_case_insensitive(cls):
        return True

    @classmethod
    def _missing_(cls, value):
        if cls.get_case_insensitive():
            condition = lambda m: str(m.name).lower() == str(value).lower()
        else:
            condition = lambda m: m.name == value
        for member in cls:
            if condition(member):
                # save to value -> member mapper
                cls._value2member_map_[value] = member
                return member

    @classmethod
    def __modify_schema__(cls, field_schema: dict):
        """
        Use enum name rather than enum value for schema used by OpenAPI.
        """
        field_schema.update(enum=[item.name for item in cls])


class PydanticObjectId(ObjectId):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, str):
            v = ObjectId(v)
        if not isinstance(v, ObjectId):
            raise ValueError('Not a valid ObjectId')
        return v

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='string')


class Framework(NamedEnum):
    TensorFlow = 0
    PyTorch = 1


class Engine(NamedEnum):
    NONE = 0
    TFS = 1
    TORCHSCRIPT = 2
    ONNX = 3
    TRT = 4
    TVM = 5
    CUSTOMIZED = 6
    PYTORCH = 7


class Task(NamedEnum):
    Image_Classification = 0
    Object_Detection = 1
    Segmentation = 2


class Metric(NamedEnum):
    acc = 0
    mAP = 1
    IoU = 2


class ModelInputFormat(NamedEnum):
    FORMAT_NONE = 0
    FORMAT_NHWC = 1
    FORMAT_NCHW = 2


class Status(NamedEnum):
    Unknown = 0
    Pass = 1
    Running = 2
    Fail = 3


@unique
class ModelStatus(NamedEnum):
    """Enumerator of model status in the lifecycle
    PUBLISHED: model published to hub and ready for conversion, profiling, and deployment
    CONVERTING: model is under conversion
    PROFILING: model is under profiling
    DEPLOYED: model is deployed as a service
    DRAFT: model is under edit
    VALIDATING: model is under quick accuracy validation
    TRAINING: model is under training
    """
    PUBLISHED = 0
    CONVERTED = 1
    PROFILING = 2
    IN_SERVICE = 3
    DRAFT = 4
    VALIDATING = 5
    TRAINING = 6


class IOShape(BaseModel):
    """Class for recording input and output shape with their data type.

    Args:
        shape (List[int]): the shape of the input or output tensor.
        dtype (DataType, type, str): The data type of the input or output tensor.
        name (str): Tensor name. Default to None.
        format (ModelInputFormat): Input format, used for TensorRT currently.
            Default to `ModelInputFormat.FORMAT_NONE`.
    """

    shape: List[int]
    dtype: DataType
    name: str
    format: ModelInputFormat = ModelInputFormat.FORMAT_NONE

    def __init__(self, **data):
        from modelci.types.type_conversion import type_to_data_type

        dtype = data.pop('dtype')
        if isinstance(dtype, str):
            try:
                # if the type name is unified python type
                dtype = type_to_data_type(eval(dtype))  # nosec
            except NameError:
                # try if the dtype is `DataType`
                dtype = DataType[dtype.upper()]
        elif isinstance(dtype, (type, int)):
            dtype = DataType(dtype)
        elif isinstance(dtype, (torch.dtype, tf.dtypes.DType, np.dtype)):
            dtype = type_to_data_type(dtype)
        elif isinstance(dtype, DataType):
            pass
        else:
            raise ValueError(
                f'data type should be an instance of `type`, type name or `DataType`, but got {type(dtype)}'
            )

        # warning if the dtype is DataType.TYPE_INVALID
        if dtype == DataType.TYPE_INVALID:
            warnings.warn('`dtype` is converted to invalid.')

        super().__init__(**data, dtype=dtype)

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def example_shape(self):
        return self.shape[1:]

    @property
    def height(self):
        if self.format == ModelInputFormat.FORMAT_NONE:
            raise ValueError('No height for shape format of `ModelInputFormat.FORMAT_NONE`.')
        if self.format == ModelInputFormat.FORMAT_NCHW:
            return self.shape[2]
        if self.format == ModelInputFormat.FORMAT_NHWC:
            return self.shape[1]

    @property
    def width(self):
        if self.format == ModelInputFormat.FORMAT_NONE:
            raise ValueError('No width for shape format of `ModelInputFormat.FORMAT_NONE`.')
        if self.format == ModelInputFormat.FORMAT_NCHW:
            return self.shape[3]
        if self.format == ModelInputFormat.FORMAT_NHWC:
            return self.shape[2]

    @property
    def channel(self):
        if self.format == ModelInputFormat.FORMAT_NONE:
            raise ValueError('No channel for shape format of `ModelInputFormat.FORMAT_NONE`.')
        if self.format == ModelInputFormat.FORMAT_NCHW:
            return self.shape[1]
        if self.format == ModelInputFormat.FORMAT_NHWC:
            return self.shape[3]

    def __str__(self):
        return '{}, dtype={}, format={}'.format(self.shape, self.dtype, self.format.name)

    class Config:
        use_enum_values = True
        schema_extra = {
            'example': {
                'shape': [-1, 3, 224, 224],
                'dtype': DataType.TYPE_FP32,
                'name': 'input0',
            }
        }
