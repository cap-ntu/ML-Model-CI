#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 10/12/2020
"""
import click

import modelci.hub.cli
from modelci.app import (start as app_start, stop as app_stop)
from modelci.utils import Logger
from modelci.utils.docker_container_manager import DockerContainerManager

logger = Logger(__name__, welcome=False)


@click.group()
@click.version_option()
def cli():
    """A complete platform for managing, converting, profiling, and deploying models as cloud services (MLaaS)"""
    pass


@cli.command()
@click.option('--gpu', default=False, type=click.BOOL, is_flag=True)
def start(gpu=False):
    """Start the ModelCI service."""
    container_conn = DockerContainerManager(enable_gpu=gpu)
    if not container_conn.start():
        container_conn.connect()
    app_start()


@cli.command()
def stop():
    """Stop the ModelCI service"""
    container_conn = DockerContainerManager()
    container_conn.stop()
    app_stop()


@cli.command()
def clean():
    """Stop the ModelCI service and remove all containers."""
    # remove all services
    container_conn = DockerContainerManager()
    container_conn.stop()
    app_stop()
    container_conn.remove_all()


cli.add_command(modelci.hub.cli.commands)
cli.add_command(modelci.hub.cli.models)

if __name__ == '__main__':
    cli()
