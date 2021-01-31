#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/27/2021

ML model structure definitions.
"""
import abc
import inspect
from enum import Enum
from typing import Optional, Union, Tuple, Dict, OrderedDict

from pydantic import BaseModel, PositiveInt, conint, PositiveFloat, Field, validator
from typing_extensions import Literal


class Operation(Enum):
    """
    Operation enum to the layer or connection. There are three kinds of operations: ``'A'`` for add the specific
    layer / connection, ``'D'`` for delete the specific layer / connection, ``M`` for modify the layer /
    connection, and ``E`` for no operation.
    """
    ADD = 'A'
    DELETE = 'D'
    MODIFY = 'M'
    EMPTY = 'E'


class LayerType(Enum):
    """
    Enum of the supported layer type. This is to hint which class of layer the provided data is converted to.
    """

    LINEAR = 'torch.nn.Linear'
    CONV_1D = 'torch.nn.Conv1d'
    CONV_2D = 'torch.nn.Conv2d'
    RELU = 'torch.nn.ReLU'
    TANH = 'torch.nn.Tanh'
    BN_1D = 'torch.nn.BatchNorm1d'
    BN_2D = 'torch.nn.BatchNorm2d'


class ModelLayer(BaseModel, abc.ABC):
    # noinspection PyUnresolvedReferences
    """
    Layer of the model structure.

    For layer attributes need to be set :code:`None`, use :code:`'null'` instead. This is for the reason of
    updated parameters with value :code:`None` will be viewed as not set. So we take special care to the
    desired :code:`None`, replacing it with :code:`'null'`.

    Attributes:
        op_ (Operation): Operation to the layer.
        type_ (LayerType): Indicates the type of this layer. This field also provides hint for :class:`pydantic`
            model conversion.
        __required_type__ (LayerType): By overriding this attributes, we can use :meth:`check_layer_type` to
            provide validation of the sub classes.
    """

    op_: Operation
    type_: LayerType

    __required_type__: LayerType

    @classmethod
    def parse_layer_obj(cls, layer_obj):
        """
        Parse from a ML layer object.

        This function will inspect the required parameters to build the layer, and try to obtain its
        parameter value from the layer object. The default parameter parser is python default
        :code:`getattr`, which assume we can get the value from the same-named attribute of the
        layer object.

        For parameter cannot parsed with default parser, set a function with the format:
        :code:`__{parameter_name}_parser__(layer_obj: Any) -> Any`.
        Has the following signature:
            Input Arguments:
            * layer_obj : Any
                The layer object to be parsed.
            Return Arguments:
            * Any
                The parsed value of the given parameter.

        TODO:
            Signature checking for __{parameter_name}_parser__
        """
        kwargs = {'op_': Operation.EMPTY, 'type_': cls.__required_type__}
        signature = inspect.signature(layer_obj.__init__)
        for param in signature.parameters:
            parser = getattr(cls, f'__{param}_parser__', lambda obj: getattr(obj, param))
            kwargs[param] = parser(layer_obj)

        return cls(**kwargs)

    @validator('type_')
    def check_layer_type(cls, layer_type: LayerType) -> LayerType:  # noqa
        """
        Checks layer type value provided is the same as the required value.
        This is to generate validator for check :code:`layer_type` field of subclasses of :class:`ModelLayer`.
        """
        if layer_type != cls.__required_type__:
            raise ValueError(f'Expected {cls.__required_type__} but got {layer_type}')
        return layer_type


class LinearLayer(ModelLayer):
    in_features: Optional[PositiveInt]
    out_features: Optional[PositiveInt]
    bias: Optional[bool]

    __required_type__ = LayerType.LINEAR

    @staticmethod
    def __bias_parser__(layer_obj):
        return layer_obj.bias is not None


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

    @staticmethod
    def __bias_parser__(layer_obj):
        return layer_obj.bias is not None


class Conv1d(_ConvNd):
    kernel_size: Optional[Union[PositiveInt, Tuple[PositiveInt]]]
    stride: Optional[Union[PositiveInt, Tuple[PositiveInt]]]

    __required_type__ = LayerType.CONV_1D


class Conv2d(_ConvNd):
    kernel_size: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]]
    stride: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]]

    __required_type__ = LayerType.CONV_2D


class ReLU(ModelLayer):
    inplace: Optional[bool]

    __required_type__ = LayerType.RELU


class Tanh(ModelLayer):
    __required_type__ = LayerType.TANH


class _BatchNorm(ModelLayer, abc.ABC):
    num_features: Optional[PositiveInt]
    eps: Optional[PositiveFloat]
    momentum: Optional[Union[PositiveFloat, Literal['null']]]
    affine: Optional[bool]
    track_running_stats: Optional[bool]


class BatchNorm1d(_BatchNorm):
    __required_type__ = LayerType.BN_1D


class BatchNorm2d(_BatchNorm):
    __required_type__ = LayerType.BN_2D


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
        ...     'fc1': LinearLayer(in_features=1024, out_features=10, type_=LayerType.LINEAR, op_=Operation.ADD),
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
        ...     'layer': {'fc': {'in_features': 1024, 'out_features': 10, 'type_': 'torch.nn.Linear', 'op_': 'A'}},
        ...     'connection': {'conv1': {'fc1': 'A'}}
        ... }
        >>> Structure.parse_obj(structure_data)
        Structure(layer={'fc': LinearLayer(in_features=1024, out_features=10, bias=None)},
        connection={'conv1': {'fc1': <Operation.ADD: 'A'>}})

    """
    layer: OrderedDict[str, _LayerType] = Field(
        default_factory=OrderedDict,
        example={'fc': {'out_features': 10, 'type_': 'torch.nn.Linear', 'op_': 'M'}}
    )
    connection: Optional[Dict[str, Dict[str, Operation]]] = Field(
        default_factory=dict,
        example={'conv1': {'fc1': 'A'}}
    )
