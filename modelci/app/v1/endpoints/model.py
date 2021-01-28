#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/20/2020
"""
from typing import List

import torch
from fastapi import APIRouter

from modelci.hub.manager import get_remote_model_weight, register_model
from modelci.persistence.service import ModelService
from modelci.types.bo import Framework, Engine, Task, IOShape, ModelVersion
from modelci.types.type_conversion import model_data_type_to_torch, type_to_data_type
from modelci.types.vo.model_structure_vo import Structure, Operation
from modelci.types.vo.model_vo import ModelDetailOut, ModelListOut, Framework as Framework_, Engine as Engine_, \
    Task as Task_
from modelci.utils.exceptions import ModelStructureError

router = APIRouter()


@router.get('/', response_model=List[ModelListOut])
def get_all_model(name: str = None, framework: Framework_ = None, engine: Engine_ = None, task: Task_ = None,
                  version: int = None):
    if framework is not None:
        framework = Framework[framework.value.upper()]
    if engine is not None:
        engine = Engine[engine.value.upper()]
    if task is not None:
        engine = Task[task.value.upper()]

    models = ModelService.get_models(name=name, framework=framework, engine=engine, task=task, version=version)
    return list(map(ModelListOut.from_bo, models))


@router.get('/{id}', response_model=ModelDetailOut)
def get_model(*, id: str):  # noqa
    model = ModelService.get_model_by_id(id)
    return ModelDetailOut.from_bo(model)


@router.get('/structure/{id}')  # TODO: add response_model
async def get_model_structure(id: str):  # noqa
    """
    Get model structure as a model structure graph (connection between layer as edge, layers as nodes)

    Arguments:
        id (str): Model object ID.
    """
    # return model DAG
    raise NotImplementedError('Method `get_model_structure` not implemented.')


@router.patch('/structure/{id}')  # TODO: add response_model
def update_model_structure_as_new(id: str, structure: Structure, dry_run: bool = False):  # noqa
    """
    TODO: Update model structure by adjusting layers (add, modify, delete) or rewiring the
        connections between layers.

    Examples:
        Fine-tune the model by modify the layer with name 'fc' (last layer). The layer
        has a changed argument out_features = 10. op_='M' indicates the operation to this layer ('fc')
        is 'Modify'. There is no changes in layer connections.
        Therefore, the structure change summary is
            [M] fc: (...) out_features=10

        >>> from collections import OrderedDict
        >>> structure_data = {
        ...     'layer': OrderedDict({'fc': {'out_features': 10, 'op_': 'M', 'type_': 'torch.nn.Linear'}})
        ... }
        >>> update_model_structure_as_new(id=..., structure=Structure.parse_obj(structure_data))

        Use original model as a feature extractor. The new model delete the last layer named 'fc', and add two
        layers as following:
            fc1: (nn.Linear) in_features=1024, out_features=512
            fc2: (nn.Linear) in_features=512, out_features=10
        The node change summary is
            [D] fc
            [A] fc1: (nn.Linear) in_features=1024, out_features=512
            [A] fc2: (nn.Linear) in_features=512, out_features=10
        Besides, we have connection changes:
            [D] conv1 -> fc
            [A] conv1 -> fc1
            [A] fc1 -> fc2

        >>>
        ... structure_data = {
        ...     'layer': {
        ...         'fc': {'_op': 'D'},
        ...         'fc1': {'in_features': 1024, 'out_features': 512, 'type_': 'torch.nn.Linear', '_op': 'A'},
        ...         'fc2': {'in_features': 512, 'out_features': 10, 'type_': 'torch.nn.Linear', '_op': 'A'},
        ...     },
        ...     'connection': {
        ...         'conv1': {'fc': 'D', 'fc1': 'A'},
        ...         'fc1': {'fc2': 'A'},
        ...     }
        ... }

    Args:
        id (str): Model object ID of the original structure.
        structure: A model structure graph indicating changed layer (node) and layer connection (edge).
        dry_run (bool): Dry run update for validation.

    Returns:

    """
    raise NotImplementedError('Method `update_model_structure_as_new` not implemented.')


@router.patch('/structure/{id}/finetune')
def update_finetune_model_as_new(id: str, updated_layer: Structure, dry_run: bool = False):  # noqa
    """
    Temporary function for finetune models. The function's functionality is overlapped with
    `update_model_structure_as_new`. Please use the `update_model_structure_as_new` in next release.

    Examples:
        Fine-tune the model by modify the layer with name 'fc' (last layer). The layer
        has a changed argument out_features = 10. op_='M' indicates the operation to this layer ('fc')
        is 'Modify'. There is no changes in layer connections.
        Therefore, the structure change summary is
            [M] fc: (...) out_features=10

        >>> from collections import OrderedDict
        >>> structure_data = {
        ...     'layer': OrderedDict({'fc': {'out_features': 10, 'op_': 'M', 'type_': 'torch.nn.Linear'}})
        ... }
        >>> update_finetune_model_as_new(id=..., updated_layer=Structure.parse_obj(structure_data))

    Args:
        id (str): ID of the model to be updated.
        updated_layer (Structure): Contains layers to be fine-tuned.
        dry_run (bool): Test run for verify if the provided parameter (i.e. model specified in `id`
            and updated layers) is valid.

    Returns:

    """
    model = ModelService.get_model_by_id(id)
    if model.engine != Engine.PYTORCH:
        raise ValueError(f'model {id} is not supported for editing. '
                         f'Currently only support model with engine={Engine_.PYTORCH}')

    # download model as local cache
    cache_path = get_remote_model_weight(model=model)
    net = torch.load(cache_path)

    for layer_name, layer_param in updated_layer.layer.items():
        layer_op = getattr(layer_param, 'op_')

        # update layer
        if layer_op == Operation.MODIFY:

            # check if the layer name exists
            if not hasattr(net, layer_name):
                raise ModelStructureError(f'Structure layer name `{layer_name}` not found in model {id}.')
            net_layer = getattr(net, layer_name)

            # check if the provided type matches the original type
            layer_type = type(net_layer)
            layer_type_provided = eval(layer_param.type_.value)
            if layer_type is not layer_type_provided:
                raise ModelStructureError(f'Expect `{layer_name}.type_` to be {layer_type}, '
                                          f'but got {layer_type_provided}')

            # get layer parameters
            layer_param_old = layer_param.parse_layer_obj(net_layer)
            layer_param_data = layer_param_old.dict(exclude_none=True, exclude={'type_', 'op_'})

            layer_param_update_data = layer_param.dict(exclude_none=True, exclude={'type_', 'op_'})
            # replace 'null' with None. See reason :class:`ModelLayer`.
            for k, v in layer_param_update_data.items():
                if v == 'null':
                    layer_param_update_data[k] = None

            # update the layer parameters
            layer_param_data.update(layer_param_update_data)
            layer = layer_type(**layer_param_data)
            setattr(net, layer_name, layer)

        elif layer_op == Operation.ADD:

            # check if the layer name not exists
            if hasattr(net, layer_name):
                raise ModelStructureError(f'Structure layer name `{layer_name}` found in model {id}.'
                                          'Operation not permitted.')
            layer_type = eval(layer_param.type_.value)
            layer_param_data = layer_param.dict(exclude_none=True, exclude={'layer_type'})
            layer = layer_type(**layer_param_data)
            setattr(net, layer_name, layer)

            # change `forward` function
            raise ValueError('Operation not permitted. Please use `update_model_structure_as_new`.')

        elif layer_op == Operation.DELETE:

            # check if the layer name exists
            if not hasattr(net, layer_name):
                raise ModelStructureError(f'Structure layer name `{layer_name}` not found in model {id}.')

            delattr(net, layer_name)

            # change `forward` function
            raise ValueError('Operation not permitted. Please use `update_model_structure_as_new`.')

    input_tensors = list()
    for input_ in model.inputs:
        input_tensor = torch.rand(1, *input_.shape[1:]).type(model_data_type_to_torch(input_.dtype))
        input_tensors.append(input_tensor)

    # parse output tensors
    output_shapes = list()
    output_tensors = net(*input_tensors)
    if not isinstance(output_tensors, (list, tuple)):
        output_tensors = (output_tensors,)
    for output_tensor in output_tensors:
        output_shape = IOShape(shape=output_tensor.shape, dtype=type_to_data_type(output_tensor.dtype))
        output_shapes.append(output_shape)

    if not dry_run:
        register_model(
            net, dataset='', metric=model.metric, task=model.task,
            inputs=model.inputs, outputs=output_shapes,
            architecture=model.name, framework=model.framework, engine=model.engine,
            version=ModelVersion(model.version.ver + 1),
            convert=False, profile=False
        )

    return True
