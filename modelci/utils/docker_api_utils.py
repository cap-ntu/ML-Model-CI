#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 10/3/2020

Docker Container API utilization.
"""
from docker.errors import ImageNotFound


def check_container_status(docker_client, name):
    """Check an existed container running status and health.

    Args:
        docker_client (docker.client.DockerClient):
        name (str): Name of the container.

    Returns:

    """
    state = docker_client.containers.get(name).attrs.get('State')
    return state is not None and state.get('Status') == 'running'


def list_containers(docker_client, filters):
    return docker_client.containers.list(all=True, filters=filters)


def get_image(docker_client, name, logger):
    """Get Docker image.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
        name (str): Image name.
        logger (modelci.utils.Logger): logger instance.

    Returns:
        docker.models.images.Image: Docker image.
    """
    try:
        image = docker_client.images.get(name)
    except ImageNotFound:
        logger.info(f'pulling {name}...')
        image = docker_client.images.pull(name)

    return image
