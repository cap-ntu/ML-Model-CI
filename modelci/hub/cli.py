#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 10/12/2020
"""
import click

from modelci.hub.init_data import export_model
from modelci.hub.manager import retrieve_model
from modelci.types.bo import Framework, Engine, ModelVersion
from modelci.ui import model_view


@click.command()
@click.argument('name', type=click.STRING, required=False)
@click.option(
    '-f', '--framework',
    type=click.Choice(['TensorFlow', 'PyTorch'], case_sensitive=False),
    help='Model framework.'
)
@click.option(
    '-e', '--engine',
    type=click.Choice(['NONE', 'TFS', 'TORCHSCRIPT', 'ONNX', 'TRT', 'TVM', 'CUSTOMIZED'], case_sensitive=False),
    help='Model serving engine.'
)
@click.option('-v', '--version', type=click.INT, help='Model version')
def models(name, framework, engine, version):
    if name:
        name = name.lower()
    if framework:
        framework = Framework[framework.upper()]
    if engine:
        engine = Engine[engine.upper()]
    if version:
        version = ModelVersion(version)
    model_list = retrieve_model(name, framework=framework, engine=engine, version=version, download=False)
    model_view([model_list])


@click.group('model')
def commands():
    """
    ModelCI hub for Manage (CURD), convert, diagnose and deploy DL models supported by industrial
    serving systems.
    """
    pass


@commands.command()
@click.option('-n', '--name', type=click.STRING, required=True, help='Model architecture name.')
@click.option(
    '-f', '--framework',
    type=click.Choice(['TensorFlow', 'PyTorch'], case_sensitive=False),
    required=True,
    help='Model framework name.'
)
@click.option(
    '--trt',
    type=click.STRING,
    is_flag=True,
    help='Flag for exporting models served by TensorRT. Please make sure you have TensorRT installed in your machine'
         'before set this flag.'
)
def export(name, framework, trt):
    """
    Export model from PyTorch hub / TensorFlow hub and try convert the model into various format for different serving
    engines.
    """
    export_model(model_name=name, framework=framework, enable_trt=trt)
