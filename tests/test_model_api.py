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
from fastapi.testclient import TestClient

from modelci.app import SERVER_PORT, SERVER_HOST
from modelci.app.main import app
from modelci.hub.publish import _download_model_from_url

client = TestClient(app)

_download_model_from_url(
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    f'{str(Path.home())}/.modelci/ResNet50/pytorch-pytorch/image_classification/1.pth'
)

def test_get_all_models():
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    assert response.json() == []


def test_publish_model():
    payload = {'convert': True, 'profile': False}
    form_data = {'architecture': 'ResNet50', 'framework': '1', 'engine': '7', 'version': '1', 'dataset': 'ImageNet',
                 'metric': "{'acc': 0.76}", 'task': '0',
                 'inputs': "[{'shape': [-1, 3, 224, 224], 'dtype': 11, 'name': 'input', 'format': 0}]",
                 'outputs': "[{'shape': [-1, 1000], 'dtype': 11, 'name': 'output', 'format': 0}]"}
    files = []
    weights_file = f'{str(Path.home())}/.modelci/ResNet50/pytorch-pytorch/image_classification/1.pth'
    files.append(("files", (weights_file, open(Path(weights_file), 'rb'), 'application/example')))
    response = requests.post(f'http://{SERVER_HOST}:{SERVER_PORT}/api/v1/model/', params=payload, data=form_data,
                             files=files)
    assert response.status_code == HTTPStatus.CREATED
    assert '"status":true' in response.text


def test_update_model():
    with requests.get(f'http://{SERVER_HOST}:{SERVER_PORT}/api/v1/model/') as r:
        model_list = r.json()
    model_id = model_list[0]["_id"]
    response = requests.patch(f'http://{SERVER_HOST}:{SERVER_PORT}/api/v1/model/{model_id}', json={'framework': 'TensorFlow'})
    assert response.status_code == HTTPStatus.OK
    assert '"framework":0' in response.text


def test_delete_model():
    with requests.get(f'http://{SERVER_HOST}:{SERVER_PORT}/api/v1/model/') as r:
        model_list = r.json()
    model_id = model_list[0]["_id"]
    response = requests.delete(f'http://{SERVER_HOST}:{SERVER_PORT}/api/v1/model/{model_id}')
    assert response.status_code == HTTPStatus.OK
    assert f'"deleted":"{model_id}"' in response.text
