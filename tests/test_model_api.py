#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 10/14/2020
"""
from http import HTTPStatus
from pathlib import Path

import requests
import torch
import torchvision

from modelci.config import app_settings

file_dir = str(Path.home() / '.modelci/ResNet50/pytorh-pytorch/image_classification')
Path(file_dir).mkdir(parents=True, exist_ok=True)
model_path = f'{file_dir}/1.pth'

torch_model = torchvision.models.resnet50(pretrained=True)
torch.save(torch_model, model_path)


def test_get_all_models():
    response = requests.get(f'{app_settings.api_v1_prefix}/model')
    assert response.status_code == HTTPStatus.OK
    assert response.json() == []


def test_publish_model():
    payload = {'convert': True, 'profile': False}
    form_data = {'architecture': 'ResNet50', 'framework': '1', 'engine': '7', 'version': '1', 'dataset': 'ImageNet',
                 'metric': "{'acc': 0.76}", 'task': '0',
                 'inputs': "[{'shape': [-1, 3, 224, 224], 'dtype': 11, 'name': 'input', 'format': 0}]",
                 'outputs': "[{'shape': [-1, 1000], 'dtype': 11, 'name': 'output', 'format': 0}]"}
    files = [("files", (model_path, open(Path(model_path), 'rb'), 'application/example'))]
    response = requests.post(f'{app_settings.api_v1_prefix}/model/', params=payload, data=form_data,
                             files=files)
    assert response.status_code == HTTPStatus.CREATED
    assert '"status":true' in response.text


def test_get_model_by_id():
    with requests.get(f'{app_settings.api_v1_prefix}/model/') as r:
        model_list = r.json()
    model_id = model_list[0]["id"]
    response = requests.get(f'{app_settings.api_v1_prefix}/model/{model_id}')
    assert response.status_code == HTTPStatus.OK
    assert model_id in response.text


def test_generate_model_graph():
    params = {'architecture': 'ResNet50', 'framework': 'PyTorch', 'engine': 'PYTORCH'}
    with requests.get(f'{app_settings.api_v1_prefix}/model/', params=params) as r:
        model_list = r.json()
    model_id = model_list[0]["id"]
    response = requests.get(f'{app_settings.api_v1_prefix}/visualizer/{model_id}')
    assert response.status_code == HTTPStatus.OK
    assert 'digraph' in response.text


def test_update_model_strcuture():
    params = {'architecture': 'ResNet50', 'framework': 'PyTorch', 'engine': 'PYTORCH'}
    with requests.get(f'{app_settings.api_v1_prefix}/model/', params=params) as r:
        model_list = r.json()
    model_id = model_list[0]["id"]
    payload = {"layer": {
        "fc": {"op_": "M", "type_": "torch.nn.Linear", "in_features": 2048, "out_features": 10, "bias": True}},
               "connection": {}}
    response = requests.patch(f'{app_settings.server_url}/api/exp/cv-tuner/finetune/{model_id}', json=payload)
    assert response.status_code == HTTPStatus.OK
    assert 'id' in response.text


def test_create_training_job():
    params = {'architecture': 'ResNet50', 'framework': 'PyTorch', 'engine': 'NONE'}
    with requests.get(f'{app_settings.api_v1_prefix}/model/', params=params) as r:
        model_list = r.json()
    model_id = model_list[0]["id"]
    payload = {"model": model_id, "data_module": {"dataset_name": "CIFAR10", "batch_size": 4},
               "min_epochs": 10, "max_epochs": 15, "optimizer_type": "Adam",
               "optimizer_property": {"betas": [0.9, 0.99], "eps": 1e-8, "weight_decay": 0, "amsgrad": False},
               "lr_scheduler_type": "StepLR", "lr_scheduler_property": {"lr": 0.01, "step_size": 30},
               "loss_function": "torch.nn.CrossEntropyLoss"}
    response = requests.post(f'{app_settings.server_url}/api/exp/train/', json=payload)
    assert response.status_code == HTTPStatus.CREATED
    assert 'id' in response.text


def test_update_model():
    with requests.get(f'{app_settings.api_v1_prefix}/model/') as r:
        model_list = r.json()
    model_id = model_list[0]["id"]
    response = requests.patch(f'{app_settings.api_v1_prefix}/model/{model_id}',
                              json={'framework': 'TensorFlow'})
    assert response.status_code == HTTPStatus.OK
    assert '"framework":0' in response.text


def test_delete_model():
    with requests.get(f'{app_settings.api_v1_prefix}/model/') as r:
        model_list = r.json()
    for model in model_list:
        model_id = model["id"]
        response = requests.delete(f'{app_settings.api_v1_prefix}/model/{model_id}')
        assert response.status_code == HTTPStatus.NO_CONTENT
