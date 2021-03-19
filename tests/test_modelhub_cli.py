# test ModelCI CLI with unitest
from pathlib import Path

import requests
from typer.testing import CliRunner

from modelci.config import app_settings
from modelci.cli.modelhub import app

runner = CliRunner()
Path(f"{str(Path.home())}/.modelci/ResNet50/pytorch-pytorch/image_classification").mkdir(parents=True, exist_ok=True)


def test_get():
    result = runner.invoke(app, [
        'get',
        'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        f'{str(Path.home())}/.modelci/ResNet50/pytorch-pytorch/image_classification/1.pth'
    ])
    assert result.exit_code == 0
    assert "model downloaded successfully" in result.stdout


def test_publish():
    result = runner.invoke(app, [
        'publish', '-f', 'example/resnet50.yml'
    ])
    assert result.exit_code == 0
    assert "\'status\': True" in result.stdout


def test_ls():
    result = runner.invoke(app, ['ls'])
    assert result.exit_code == 0


def test_detail():
    with requests.get(f'{app_settings.api_v1_prefix}/model/') as r:
        model_list = r.json()
    model_id = model_list[0]["_id"]
    result = runner.invoke(app, ['detail', model_id])
    assert result.exit_code == 0


def test_update():
    with requests.get(f'{app_settings.api_v1_prefix}/model/') as r:
        model_list = r.json()
    model_id = model_list[0]["_id"]
    result = runner.invoke(app, ['update', model_id, '--framework', 'TensorFlow'])
    assert result.exit_code == 0


def test_delete():
    with requests.get(f'{app_settings.api_v1_prefix}/model/') as r:
        model_list = r.json()
    model_id = model_list[0]["_id"]
    result = runner.invoke(app, ['delete', model_id])
    assert result.exit_code == 0
    assert f'\'deleted\': \'{model_id}\'' in result.output
