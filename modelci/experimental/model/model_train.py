#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/31/2021

ML model training parameters.
"""
import abc
from enum import Enum
from typing import Optional, Union, Tuple, List

from pydantic import BaseModel, PositiveInt, PositiveFloat, root_validator, validator


class DataModuleProperty(BaseModel):
    dataset: str
    batch_size: int
    num_workers: Optional[int] = 1
    data_dir: Optional[str]


class OptimizerType(Enum):
    SGD = 'SGD'
    ADAGRAD = 'Adagrad'
    ADAM = 'Adam'


# noinspection PyUnresolvedReferences
class OptimizerPropertyBase(BaseModel, abc.ABC):
    """
    Base class of optimizer property. It provides parameters to initialize optimizer.

    Attributes:
        __required_type__ (OptimizerType): By overriding this attribute, the subclass type can be automatically
            determined by :meth:`check_required_type`.
    """
    lr: Optional[PositiveFloat]

    __required_type__: OptimizerType

    @root_validator(pre=True)
    def check_required_type(cls, values):  # noqa
        required_type = values.get('__required_type__')
        if isinstance(required_type, str):
            required_type = OptimizerType(required_type)
        if required_type != cls.__required_type__:
            raise ValueError(f'Given type {required_type}, required {cls.__required_type__}')
        return values


class SGDProperty(OptimizerPropertyBase):
    momentum: Optional[PositiveFloat]

    __required_type__ = OptimizerType.SGD


class AdagradProperty(OptimizerPropertyBase):
    lr_decay: Optional[PositiveFloat]
    weight_decay: Optional[PositiveFloat]
    eps: Optional[PositiveFloat]


class AdamProperty(OptimizerPropertyBase):
    betas: Optional[Tuple[PositiveFloat, PositiveFloat]]
    eps: Optional[Tuple[PositiveFloat]]
    weight_decay: Optional[PositiveFloat]
    amsgrad: Optional[bool]


_OptimizerProperty = Union[SGDProperty, AdagradProperty, AdamProperty]


class LRSchedulerType(Enum):
    STEP_LR = 'StepLR'
    EXPONENTIAL_LR = 'ExponentialLR'


# noinspection PyUnresolvedReferences
class LRSchedulerPropertyBase(BaseModel, abc.ABC):
    """
    Base class of learning rate scheduler property. It provides parameters to initialize LR scheduler.

    Attributes:
        __required_type__ (LRSchedulerType): By overriding this attribute, the subclass type can be automatically
            determined by :meth:`check_required_type`.
    """
    last_epoch: Optional[int]
    verbose: Optional[bool]

    __required_type__: LRSchedulerType

    @root_validator(pre=True)
    def check_required_type(cls, values):  # noqa
        required_type = values.get('__required_type__')
        if isinstance(required_type, str):
            required_type = OptimizerType(required_type)
        if required_type != cls.__required_type__:
            raise ValueError(f'Given type {required_type}, required {cls.__required_type__}')
        return values


class StepLRProperty(LRSchedulerPropertyBase):
    step_size: PositiveInt
    gamma: Optional[PositiveFloat]


class MultiStepLRProperty(LRSchedulerPropertyBase):
    milestones: List[PositiveInt]
    gamma: Optional[PositiveFloat]

    @validator('milestones')
    def check_list_increasing(cls, v):
        if not all(i < j for i, j in zip(v, v[1:])):
            raise ValueError(f'List {v} is not strictly increasing.')


class ExponentialLRProperty(LRSchedulerPropertyBase):
    gamma: Optional[PositiveFloat]


_LRSchedulerProperty = Union[StepLRProperty, MultiStepLRProperty, ExponentialLRProperty]


class LossFunctionType(Enum):
    L1_Loss = 'L1Loss'
    MSE_Loss = 'MSELoss'
    CROSS_ENTROPY_LOSS = 'CrossEntropyLoss'


class TrainingJob(BaseModel):
    id: str
    data_module: DataModuleProperty
    min_epochs: Optional[PositiveInt]
    max_epochs: PositiveInt
    optimizer_type: OptimizerType
    optimizer_property: _OptimizerProperty
    lr_scheduler_type: LRSchedulerType
    lr_scheduler_property: _LRSchedulerProperty
    loss_function: LossFunctionType

    @root_validator(pre=True)
    def optimizer_type_inject(cls, values):  # noqa
        """
        Inject type for `optimizer_property` based on the value provided in `optimizer_type`.
        """
        test_type, test_prop = values.get('optimizer_type'), values.get('optimizer_property')
        if isinstance(test_prop, dict):
            test_prop['__required_type__'] = test_type
        elif isinstance(test_prop, OptimizerPropertyBase):
            if test_prop.__required_type__.value != test_type:
                raise TypeError(f'`optimizer_property` has incorrect type {type(test_prop)} as '
                                f'defined in `optimizer_type`: {test_type}.')
        else:
            raise TypeError(
                f'Cannot parse type, expected one of [`dict`, `OptimizerProperty`], got {type(test_prop)}.'
            )

        return values
