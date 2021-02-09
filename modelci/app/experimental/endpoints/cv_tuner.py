#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: yuanmingleee
Email: 
Date: 1/29/2021
"""

import torch
from fastapi import APIRouter


from modelci.experimental.model.model_structure import Structure, Operation
from modelci.hub.manager import register_model, get_remote_model_weight
from modelci.persistence.service import ModelService
from modelci.types.bo import ModelVersion, Engine, IOShape, ModelStatus
from modelci.types.type_conversion import model_data_type_to_torch, type_to_data_type
from modelci.utils.exceptions import ModelStructureError

router = APIRouter()


@router.patch('/finetune/{id}')
def update_finetune_model_as_new(id: str, updated_layer: Structure, dry_run: bool = False):  # noqa
    """
    Temporary function for finetune CV models. The function's functionality is overlapped with
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
    if len(updated_layer.layer.items()) == 0:
        return True
    model = ModelService.get_model_by_id(id)
    if model.engine != Engine.PYTORCH:
        raise ValueError(f'model {id} is not supported for editing. '
                         f'Currently only support model with engine=PYTORCH')
    # download model as local cache
    cache_path = get_remote_model_weight(model=model)
    net = torch.load(cache_path)

    for layer_name, layer_param in updated_layer.layer.items():
        layer_op = getattr(layer_param, 'op_')

        # update layer
        if layer_op == Operation.MODIFY:

            # check if the layer name exists
            # TODO check if layer path exists eg."layer1.0.conv1"
            if not hasattr(net, layer_name):
                raise ModelStructureError(f'Structure layer name `{layer_name}` not found in model {id}.')
            net_layer = getattr(net, layer_name)

            # check if the provided type matches the original type
            layer_type = type(net_layer)
            layer_type_provided = eval(layer_param.type_.value)  # nosec
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

        else:
            # if layer_op is Operation.ADD,
            #     1. check if the layer name not exists
            #     2. add a layer
            #     3. change the `forward` function according to the connections
            # if layer_op is Operation.DELETE,
            #     1. check if the layer exists
            #     2. delete the layer
            #     3. change the `forward` function
            raise ValueError('Operation not permitted. Please use `update_model_structure_as_new`.')

    input_tensors = list()
    bs = 1
    for input_ in model.inputs:
        input_tensor = torch.rand(bs, *input_.shape[1:]).type(model_data_type_to_torch(input_.dtype))
        input_tensors.append(input_tensor)

    # parse output tensors
    output_shapes = list()
    output_tensors = net(*input_tensors)
    if not isinstance(output_tensors, (list, tuple)):
        output_tensors = (output_tensors,)
    for output_tensor in output_tensors:
        output_shape = IOShape(shape=[bs, *output_tensor.shape[1:]], dtype=type_to_data_type(output_tensor.dtype))
        output_shapes.append(output_shape)

    if not dry_run:
        # TODO return validation result for dry_run mode
        # TODO apply Semantic Versioning https://semver.org/
        # TODO reslove duplicate model version problem in a more efficient way
        version = ModelVersion(model.version.ver + 1)
        previous_models = ModelService.get_models(
                name=model.name,
                task=model.task,
                framework=model.framework,
                engine=Engine.NONE
        )
        if len(previous_models):
            last_version = max(previous_models, key=lambda k: k.version.ver).version.ver
            version = ModelVersion(last_version + 1)

        register_model(
            net,
            dataset='',
            metric={key: 0 for key in model.metric.keys()},
            task=model.task,
            inputs=model.inputs,
            outputs=output_shapes,
            architecture=model.name,
            framework=model.framework,
            engine=Engine.NONE,
            model_status=[ModelStatus.DRAFT],
            version=version,
            convert=False, profile=False
        )

        model_bo = ModelService.get_models(
            name=model.name,
            task=model.task,
            framework=model.framework,
            engine=Engine.NONE,
            version=version
        )[0]

        return {'id' : model_bo.id}
