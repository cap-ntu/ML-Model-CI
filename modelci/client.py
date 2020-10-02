#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 10/2/2020
"""
import subprocess
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


def start_cadvisor(docker_client, name='modelci.cadvisor', gpu=False, port=8080):
    """Start cAdvisor service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
    """
    try:
        docker_client.images.get('google/cadvisor:latest')
    except ImageNotFound:
        docker_client.images.pull('google/cadvisor:latest')

    try:
        container = docker_client.containers.get(name)
        print(f'Already exist, found container name={name}.')

        container.start()
        print('Stared')
        return
    except NotFound:
        pass

    if gpu:
        # find libnvidia-ml.so.1
        cache_file = Path('/tmp/libnvidia-ml.cache')
        if cache_file.exists():
            with open(cache_file) as f:
                lib_path = f.read().strip()
        else:
            args1 = ('locate', 'libnvidia-ml.so.1')
            args2 = ('grep', '-v', 'lib32')
            args3 = ('head', '-1')
            locate = subprocess.Popen(args1, stdout=subprocess.PIPE)
            grep = subprocess.Popen(args2, stdin=locate.stdout, stdout=subprocess.PIPE)
            locate.wait()
            grep.wait()

            lib_path = subprocess.check_output(args3, stdin=grep.stdout, universal_newlines=True).strip()

            # save to cache
            with open(cache_file, 'w') as f:
                f.write(lib_path)

        print('starting cAdvisor...')
        docker_client.containers.run(
            'google/cadvisor:latest', name=name, ports={'8080/tcp': port}, detach=True, privileged=True,
            environment={'LD_LIBRARY_PATH': str(Path(lib_path).parent)},
            volumes={
                lib_path: {'bind': lib_path},
                '/': {'bind': '/rootfs', 'mode': 'ro'},
                '/var/run': {'bind': '/var/run', 'mode': 'rw'},
                '/sys': {'bind': '/sys', 'mode': 'ro'},
                '/var/lib/docker': {'bind': '/var/lib/docker', 'mode': 'ro'},
            }
        )
    else:
        docker_client.containers.run(
            'google/cadvisor:latest', name=name, ports={'8080/tcp': port}, detach=True, privileged=True,
            volumes={
                '/': {'bind': '/rootfs', 'mode': 'ro'},
                '/var/run': {'bind': '/var/run', 'mode': 'rw'},
                '/sys': {'bind': '/sys', 'mode': 'ro'},
                '/var/lib/docker': {'bind': '/var/lib/docker', 'mode': 'ro'},
            }
        )


def stop_cadvisor(docker_client, name='modelci.cadvisor'):
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


def start_node_exporter(docker_client, name='modelci.{}-exporter', port=9400):
    """Start cAdvisor service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
        name (str): Name template for node exporter. Default to be 'modelci.{}-exporter'.
    """
    try:
        docker_client.images.get('bgbiao/dcgm-exporter:latest')
    except ImageNotFound:
        docker_client.images.pull('bgbiao/dcgm-exporter:latest')

    try:
        docker_client.images.get('bgbiao/gpu-metrics-exporter:latest')
    except ImageNotFound:
        docker_client.images.pull('bgbiao/gpu-metrics-exporter:latest')

    # start dcgm-exporter
    dcgm_container = docker_client.containers.run(
        'bgbiao/dcgm-exporter', detach=True, runtime='nvidia', name=name.format('dcgm')
    )

    # start gpu-metric-exporter
    docker_client.containers.run(
        'bgbiao/gpu-metrics-exporter', detach=True, privileged=True, name=name.format('gpu-metric'),
        ports={'9400/tcp': port}, volumes_from=[dcgm_container.id]
    )


def stop_node_exporter(docker_client, name='modelci.{}-exporter'):
    for sub_name in ['dcgm', 'gpu-metric']:
        try:
            container = docker_client.containers.get(name.format(sub_name))
            container.stop()
            print(f'{name.format(sub_name)} stopped')
        except NotFound:
            print(f'Service not started: container {name.format(sub_name)} not found')


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
    stop_node_exporter(c)
