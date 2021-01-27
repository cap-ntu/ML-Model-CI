#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/27/2021
"""
import abc
from enum import Enum
from typing import Optional, Union, Tuple, Dict, OrderedDict

from pydantic import BaseModel, PositiveInt, conint, PositiveFloat, Field
from typing_extensions import Literal


class Operation(Enum):
    """
    Operation enum to the layer or connection. There are three kinds of operations: ``'A'`` for add the specific
    layer / connection, ``'D'`` for delete the specific layer / connection, and ``M`` for modify the layer /
    connection.
    """
    ADD = 'A'
    DELETE = 'D'
    MODIFY = 'M'


class LayerType(Enum):
    """
    Enum of the supported layer type. This is to hint which class of layer the provided data is converted to.
    """

    LINEAR = 'nn.Linear'
    CONV_1D = 'nn.Conv1d'
    CONV_2D = 'nn.Conv2d'
    RELU = 'nn.ReLU'
    TANH = 'nn.Tanh'
    BN_1D = 'nn.BatchNorm1d'
    BN_2D = 'nn.BatchNorm2d'


class ModelLayer(BaseModel, abc.ABC):
    # noinspection PyUnresolvedReferences
    """
    Layer of the model structure.

    Attributes:
        _op (Operation): Operation to the layer.
        _type (LayerType): Indicates the type of this layer. This field also provides hint for :class:`pydantic`
            model conversion.
    """

    _op: Operation
    _type: LayerType


class LinearLayer(ModelLayer):
    _type = Field(LayerType.LINEAR, const=True)
    in_features: Optional[PositiveInt]
    out_features: Optional[PositiveInt]
    bias: Optional[bool]


class _ConvNd(ModelLayer, abc.ABC):
    in_channels: Optional[PositiveInt]
    out_channels: Optional[PositiveInt]
    kernel_size: Optional[Union[PositiveInt, Tuple[PositiveInt, ...]]]
    stride: Optional[Union[PositiveInt, Tuple[PositiveInt, ...]]]
    padding: conint(ge=0)
    dilation: PositiveInt
    groups: PositiveInt
    bias: bool
    padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular']


class Conv1d(_ConvNd):
    _type = Field(LayerType.CONV_1D, const=True)
    kernel_size: Optional[Union[PositiveInt, Tuple[PositiveInt]]]
    stride: Optional[Union[PositiveInt, Tuple[PositiveInt]]]


class Conv2d(_ConvNd):
    _type = Field(LayerType.CONV_2D, const=True)
    kernel_size: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]]
    stride: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]]


class ReLU(ModelLayer):
    _type = Field(LayerType.RELU, const=True)
    inplace: Optional[bool]


class Tanh(ModelLayer):
    _type = Field(LayerType.TANH, const=True)


class _BatchNorm(ModelLayer, abc.ABC):
    num_features: Optional[PositiveInt]
    eps: Optional[PositiveFloat]
    momentum: Optional[Union[PositiveFloat, Literal['null']]]
    affine: Optional[bool]
    track_running_stats: Optional[bool]


class BatchNorm1d(_BatchNorm):
    _type = Field(LayerType.BN_1D, const=True)


class BatchNorm2d(_BatchNorm):
    _type = Field(LayerType.BN_2D, const=True)


_LayerType = Union[LinearLayer, Conv1d, Conv2d, ReLU, Tanh, BatchNorm1d, BatchNorm2d]


class Structure(BaseModel):
    # noinspection PyUnresolvedReferences
    """
    Indicate a ML model structure using a graph data structure.
    :attr:`layer` is the graph node, representing a layer of the model. :attr:`connection` is the graph edge,
    representing which two layers are connected, and the directions of tensor pass.

    Attributes:
        layer (OrderedDict[str, _LayerType]): Layer mapping, the key is layer name, and the value is layer
            attributes. See :class:`ModelLayer` for reference.
        connection (Optional[Dict[str, Dict[str, Operation]]]): The connection (:attr:`connection`) maps
            the starting layer name, to the ending layer name with a connection operation.

    Examples::

        >>> from collections import OrderedDict
        >>> # add a nn.Linear layer named 'fc1' with in_features=1024, out_features=10
        >>> layer_mapping = OrderedDict({
        ...     'fc1': LinearLayer(in_features=1024, out_features=10, _type=LayerType.LINEAR, _op=Operation.ADD),
        ... })
        >>> # connection example for add connection from 'conv1' to 'fc1'
        >>> connection_mapping = {'conv1': {'fc1': Operation.ADD}}
        >>> struct = Structure(layer=layer_mapping, connection=connection_mapping)
        >>> print(struct)
        layer={'fc1': LinearLayer(in_features=1024, out_features=10, bias=None)}
        connection={'conv1': {'fc1': <Operation.ADD: 'A'>}}
        >>> # Other than using the model object, we can pass in a plain dictionary,
        ... # and utilize `Structure.parse_obj`.
        >>> structure_data = {
        ...     'layer': {'fc': {'in_features': 1024, 'out_features': 10, '_type': 'nn.Linear', '_op': 'A'}},
        ...     'connection': {'conv1': {'fc1': 'A'}}
        ... }
        >>> Structure.parse_obj(structure_data)
        Structure(layer={'fc': LinearLayer(in_features=1024, out_features=10, bias=None)},
        connection={'conv1': {'fc1': <Operation.ADD: 'A'>}})

    """
    layer: OrderedDict[str, _LayerType] = Field(default_factory=OrderedDict)
    connection: Optional[Dict[str, Dict[str, Operation]]] = Field(default_factory=dict)
