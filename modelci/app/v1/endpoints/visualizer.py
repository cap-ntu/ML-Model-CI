#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Jiang Shanshan
Email: univeroner@gmail.com
Date: 2021/1/15

"""
import io
import tempfile
from pathlib import Path

import onnx
import torch
from fastapi import APIRouter

from modelci.experimental.visualizer.onnx_visualizer import visualize_model
from modelci.hub.converter import convert
from modelci.persistence.service_ import get_by_id, get_models
from modelci.types.models import Engine, Graph

router = APIRouter()


@router.get('/{id}', response_model=Graph)
def generate_model_graph(*, id: str):  # noqa
    model = get_by_id(id)
    graph = None
    if model.engine == Engine.PYTORCH:
        result = get_models(architecture=model.architecture, framework=model.framework, engine=Engine.ONNX, task=model.task, version=model.version)
        if len(result):
            onnx_model = onnx.load(io.BytesIO(bytes(result[0].weight)))
        else:
            pytorch_model = torch.load(io.BytesIO(bytes(model.weight)))
            onnx_path = Path(tempfile.gettempdir() + '/tmp.onnx')
            convert(pytorch_model, 'pytorch', 'onnx', save_path=onnx_path, inputs=model.inputs, outputs=model.outputs, optimize=False)
            onnx_model = onnx.load(onnx_path)
        graph = visualize_model(onnx_model)
    return graph
