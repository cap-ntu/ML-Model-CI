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

logger = Logger(__name__, welcome=False)

MONGO_CONTAINER_NAME = 'modelci.mongo'
MONGO_INITDB_ROOT_USERNAME = 'admin'
MONGO_INITDB_ROOT_PASSWORD = 'admin'
MONGO_INITDB_USERNAME = 'modelci'
MONGO_INITDB_PASSWORD = 'modelci@2020'
MONGO_INITDB_DATABASE = 'modelci'

CADVISOR_CONTAINER_NAME = 'modelci.cadvisor'

GDCM_EXPORTER_CONTAINER_NAME = 'modelci.gdcm-exporter'
GPU_METRICS_EXPORTER_CONTAINER_NAME = 'modelci.gpu-metrics-exporter'


def start():
    """Start the ModelCI service."""
    docker_client = docker.from_env()
    download_serving_containers(docker_client)

    if docker_client.containers.list(filters={'name': MONGO_CONTAINER_NAME, 'status': 'running'}):
        logger.info('MongoDB already started.')
    else:
        start_mongodb(docker_client)

    if docker_client.containers.list(filters={'name': CADVISOR_CONTAINER_NAME, 'status': 'running'}):
        logger.info('cAdvisor already started.')
    else:
        start_cadvisor(docker_client)

    if docker_client.containers.list(filters={'name': GDCM_EXPORTER_CONTAINER_NAME, 'status': 'running'}) and \
            docker_client.containers.list(filters={'name': GPU_METRICS_EXPORTER_CONTAINER_NAME, 'status': 'running'}):
        logger.info('Node exporter already started.')
    else:
        start_node_exporter(docker_client)


def stop():
    """Stop the ModelCI service"""
    docker_client = docker.from_env()
    stop_mongodb(docker_client)
    stop_cadvisor(docker_client)
    stop_node_exporter(docker_client)


def start_mongodb(
        docker_client,
        name=MONGO_CONTAINER_NAME,
        root_user=MONGO_INITDB_ROOT_USERNAME,
        user=MONGO_INITDB_USERNAME,
        modelci_db=MONGO_INITDB_DATABASE
):
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
        logger.info(f'Already exist, found container name={name}.')
        container.start()
        logger.info(f'{name} stared')
        return
    except NotFound:
        pass

    init_sh_path = str(Path(__file__).parent.absolute() / 'init-mongo.sh')

    docker_client.containers.run(
        'mongo', detach=True, ports={'27017/tcp': '27017'}, name=name,
        volumes={init_sh_path: {'bind': '/docker-entrypoint-initdb.d/init-mongo.sh', 'mode': 'ro'}},
        environment={
            'MONGO_INITDB_ROOT_USERNAME': root_user,
            'MONGO_INITDB_ROOT_PASSWORD': MONGO_INITDB_ROOT_PASSWORD,  # TODO: to be replaced
            'MONGO_INITDB_USERNAME': user,
            'MONGO_INITDB_PASSWORD': MONGO_INITDB_PASSWORD,  # TODO: to be replaced
            'MONGO_INITDB_DATABASE': modelci_db,
        }
    )
    logger.info(f'{name} stared')


def stop_mongodb(docker_client, name=MONGO_CONTAINER_NAME):
    """Stop Mongo DB service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
    """
    try:
        container = docker_client.containers.get(name)
        container.stop()
        logger.info(f'{name} stopped')
    except NotFound:
        logger.info(f'Service not started: container {name} not found')


def start_cadvisor(docker_client, name=CADVISOR_CONTAINER_NAME, gpu=False, port=8080):
    """Start cAdvisor service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
    """
    try:
        container = docker_client.containers.get(name)
        logger.info(f'Already exist, found container name={name}.')
        container.start()
        logger.info(f'{name} stared')
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
    logger.info(f'{name} started.')


def stop_cadvisor(docker_client, name=CADVISOR_CONTAINER_NAME):
    """Stop Mongo DB service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
    """
    try:
        container = docker_client.containers.get(name)
        container.stop()
        logger.info(f'{name} stopped')
    except NotFound:
        logger.info(f'Service not started: container {name} not found')


def start_node_exporter(
        docker_client,
        dcgm_name=GDCM_EXPORTER_CONTAINER_NAME,
        gpu_metrics_name=GPU_METRICS_EXPORTER_CONTAINER_NAME,
        port=9400
):
    """Start cAdvisor service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
        dcgm_name (str): Container name of dcgm-exporter. Default to `GDCM_EXPORTER_CONTAINER_NAME`.
        gpu_metrics_name (str): Container name of gpu-metrics-exporter. Default to
            `GPU_METRICS_EXPORTER_CONTAINER_NAME`.
    """
    # start dcgm-exporter
    try:
        dcgm_container = docker_client.containers.get(dcgm_name)
        logger.info(f'Already exist, found container name={dcgm_name}.')
        dcgm_container.start()
    except NotFound:
        dcgm_container = docker_client.containers.run(
            'bgbiao/dcgm-exporter', detach=True, runtime='nvidia', name=dcgm_name
        )
    logger.info(f'{dcgm_name} stared')

    # start gpu-metric-exporter
    try:
        container = docker_client.containers.get(gpu_metrics_name)
        logger.info(f'Already exist, found container name={gpu_metrics_name}.')
        container.start()
    except NotFound:
        docker_client.containers.run(
            'bgbiao/gpu-metrics-exporter', detach=True, privileged=True, name=gpu_metrics_name,
            ports={'9400/tcp': port}, volumes_from=[dcgm_container.id]
        )
    logger.info(f'{gpu_metrics_name} stared')


def stop_node_exporter(
        docker_client,
        dcgm_name=GDCM_EXPORTER_CONTAINER_NAME,
        gpu_metrics_name=GPU_METRICS_EXPORTER_CONTAINER_NAME
):
    for name in [dcgm_name, gpu_metrics_name]:
        try:
            container = docker_client.containers.get(name)
            container.stop()
            logger.info(f'{name} stopped')
        except NotFound:
            logger.info(f'Service not started: container {name} not found')


def download_serving_containers(docker_client):
    images = [
        'mlmodelci/pytorch-serving:latest',
        'mlmodelci/pytorch-serving:latest-gpu',
        'mlmodelci/onnx-serving:latest',
        'mlmodelci/onnx-serving:latest-gpu',
        'tensorflow/serving:2.1.0',
        'tensorflow/serving:2.1.0-gpu',
        'nvcr.io/nvidia/tensorrtserver:19.10-py3',
        'mongo:latest',
        'google/cadvisor:latest',
        'bgbiao/dcgm-exporter:latest',
        'bgbiao/gpu-metrics-exporter:latest',
    ]

    for image in images:
        try:
            docker_client.images.get(image)
        except ImageNotFound:
            logger.info(f'pulling {image}...')
            docker_client.images.pull(image)


if __name__ == '__main__':
    # start()
    stop()
