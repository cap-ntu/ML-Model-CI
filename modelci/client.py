#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 10/2/2020
"""
from pathlib import Path

import docker
from docker.errors import ImageNotFound, NotFound

from modelci.utils import Logger


class ModelCIConnection(object):
    def __init__(self):
        self.docker_client = docker.from_env()
        self.docker_client.secrets.create()
        self.logger = Logger(__name__, welcome=False)

    @staticmethod
    def from_env():
        return ModelCIConnection()

    def start(self):
        """Start the ModelCI service."""
        # check if the MongoDB service has started
        containers = self.docker_client.containers.list(filters={'name': 'modelci.mongo'})


def start_mongodb(docker_client, name='modelci.mongo', root_user='admin', user='modelci', modelci_db='modelci'):
    """Start Mongo DB service.
    From https://stackoverflow.com/a/53522699/13173608.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
    """

    try:
        docker_client.images.get('mongo')
    except ImageNotFound:
        docker_client.images.pull('mongo')

    try:
        container = docker_client.containers.get(name)
        print(f'Already exist, found container name={name}.')

        container.start()
        print('Stared')
        return
    except NotFound:
        pass

    init_sh_path = str(Path(__file__).parent.absolute() / 'init-mongo.sh')

    docker_client.containers.run(
        'mongo', detach=True, ports={'27017/tcp': '27017'}, name=name,
        volumes={init_sh_path: {'bind': '/docker-entrypoint-initdb.d/init-mongo.sh', 'mode': 'ro'}},
        environment={
            'MONGO_INITDB_ROOT_USERNAME': root_user,
            'MONGO_INITDB_ROOT_PASSWORD': 'admin',  # TODO: to be replaced
            'MONGO_INITDB_USERNAME': user,
            'MONGO_INITDB_PASSWORD': 'modelci@2020',  # TODO: to be replaced
            'MONGO_INITDB_DATABASE': modelci_db,
        }
    )


def stop_mongodb(docker_client, name='modelci.mongo'):
    """Stop Mongo DB service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
    """
    try:
        container = docker_client.containers.get(name)
        container.stop()
        print(f'Stopped')
    except NotFound:
        print(f'Service not started: container {name} not found')


def download_serving_containers(docker_client):
    images = [
        'mlmodelci/pytorch-serving:latest',
        'mlmodelci/pytorch-serving:latest-gpu',
        'mlmodelci/onnx-serving:latest',
        'mlmodelci/onnx-serving:latest-gpu',
        'tensorflow/serving:2.1.0',
        'tensorflow/serving:2.1.0-gpu',
        'nvcr.io/nvidia/tensorrtserver:19.10-py3',
    ]

    for image in images:
        try:
            docker_client.images.get('mongo')
        except ImageNotFound:
            docker_client.images.pull(image)


if __name__ == '__main__':
    c = docker.from_env()
    start_mongodb(c)
