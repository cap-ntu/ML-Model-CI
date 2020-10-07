#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 10/2/2020
"""
import os
import random
import signal
import subprocess
import sys
from pathlib import Path

import docker

from modelci.app.config import SERVER_HOST, SERVER_PORT
from modelci.config import (
    MONGO_PORT, MONGO_USERNAME, MONGO_PASSWORD, MONGO_DB, MONGO_CONTAINER_LABEL,
    CADVISOR_CONTAINER_LABEL, CADVISOR_PORT,
    DCGM_EXPORTER_CONTAINER_LABEL, GPU_METRICS_EXPORTER_CONTAINER_LABEL, NODE_EXPORTER_PORT,
)
from modelci.utils import Logger
from modelci.utils.docker_api_utils import list_containers, get_image, check_container_status

logger = Logger(__name__, welcome=False)

backend_process = None


def start():
    """Start the ModelCI service."""
    docker_client = docker.from_env()
    download_serving_containers(docker_client)

    if list_containers(docker_client, filters={'label': MONGO_CONTAINER_LABEL, 'status': 'running'}):
        logger.info('MongoDB already started.')
    else:
        start_mongodb(docker_client)

    if list_containers(docker_client, filters={'label': CADVISOR_CONTAINER_LABEL, 'status': 'running'}):
        logger.info('cAdvisor already started.')
    else:
        start_cadvisor(docker_client)

    if list_containers(docker_client, filters={'label': DCGM_EXPORTER_CONTAINER_LABEL, 'status': 'running'}) and \
            list_containers(
                docker_client,
                filters={'label': GPU_METRICS_EXPORTER_CONTAINER_LABEL, 'status': 'running'}
            ):
        logger.info('Node exporter already started.')
    else:
        start_node_exporter(docker_client)

    start_fastapi_backend()


def stop():
    """Stop the ModelCI service"""
    docker_client = docker.from_env()
    stop_mongodb(docker_client)
    stop_cadvisor(docker_client)
    stop_node_exporter(docker_client)
    stop_fastapi_backend()


def _check_service_started(docker_client, label):
    containers = list_containers(docker_client, filters={'label': [label]})
    if containers:
        name = containers[0].name
        # container exist
        if check_container_status(docker_client, name):
            # container is running
            logger.info(f'Already running, found container name={name}.')
        else:
            containers[0].start()
            logger.info(f'Already exist, found container name={name}. Service started.')
        return name


def _check_process_running(pid_file):
    if Path(pid_file).exists():
        with open(pid_file) as f:
            pid = int(f.read())
        try:
            os.killpg(os.getpgid(pid), signal.ITIMER_REAL)
            logger.info(f'Already running, found PID={pid}.')
            return pid
        except OSError:
            pass
    return False


def _stop_service(docker_client, label):
    containers = list_containers(docker_client, filters={'label': [label]})
    if containers:
        name = containers[0].name
        # container exist
        if check_container_status(docker_client, name):
            # container is running
            containers[0].stop()
            logger.info(f'{name} stopped')
        else:
            logger.info('Service not started.')
    else:
        logger.info(f'Container with label: {label} not found')


def start_mongodb(docker_client, port=MONGO_PORT):
    """Start Mongo DB service.
    From https://stackoverflow.com/a/53522699/13173608.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
        port (int): Port for MongoDB service.
    """

    mongo_name = _check_service_started(docker_client, label=MONGO_CONTAINER_LABEL)

    if mongo_name is None:
        mongo_name = f'mongo-{random.randint(0, 100000)}'

        init_sh_path = str(Path(__file__).parent.absolute() / 'init-mongo.sh')

        docker_client.containers.run(
            'mongo', detach=True, ports={'27017/tcp': port}, name=mongo_name,
            volumes={init_sh_path: {'bind': '/docker-entrypoint-initdb.d/init-mongo.sh', 'mode': 'ro'}},
            environment={
                'MONGO_INITDB_USERNAME': MONGO_USERNAME,
                'MONGO_INITDB_PASSWORD': MONGO_PASSWORD,
                'MONGO_INITDB_DATABASE': MONGO_DB,
            },
            labels=[MONGO_CONTAINER_LABEL]
        )

    check_container_status(docker_client, name=mongo_name)
    logger.info(f'Container name={mongo_name} stared')


def stop_mongodb(docker_client):
    """Stop Mongo DB service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
    """
    _stop_service(docker_client, MONGO_CONTAINER_LABEL)


def start_cadvisor(docker_client, gpu=False, port=CADVISOR_PORT):
    """Start cAdvisor service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
        gpu (bool): Flag for enable GPU.
        port (int): Port for cAdvisor service.
    """
    cadvisor_name = _check_service_started(docker_client, label=CADVISOR_CONTAINER_LABEL)

    if cadvisor_name is None:
        cadvisor_name = f'cadvisor-{random.randint(0, 100000)}'

        volumes = {
            '/': {'bind': '/rootfs', 'mode': 'ro'},
            '/var/run': {'bind': '/var/run', 'mode': 'rw'},
            '/sys': {'bind': '/sys', 'mode': 'ro'},
            '/var/lib/docker': {'bind': '/var/lib/docker', 'mode': 'ro'},
        }

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
                'google/cadvisor:latest', name=cadvisor_name, ports={'8080/tcp': port}, detach=True, privileged=True,
                environment={'LD_LIBRARY_PATH': str(Path(lib_path).parent)},
                volumes={lib_path: {'bind': lib_path}, **volumes}, labels=[CADVISOR_CONTAINER_LABEL]
            )
        else:
            docker_client.containers.run(
                'google/cadvisor:latest', name=cadvisor_name, ports={'8080/tcp': port}, detach=True, privileged=True,
                volumes=volumes, labels=[CADVISOR_CONTAINER_LABEL]
            )

    check_container_status(docker_client, name=cadvisor_name)
    logger.info(f'Container name={cadvisor_name} started.')


def stop_cadvisor(docker_client):
    """Stop Mongo DB service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
    """
    _stop_service(docker_client, CADVISOR_CONTAINER_LABEL)


def start_node_exporter(
        docker_client,
        port=NODE_EXPORTER_PORT,
):
    """Start node exporter service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
        port (int): Port for node exporter service.
    """

    dcgm_name = _check_service_started(docker_client, label=DCGM_EXPORTER_CONTAINER_LABEL)
    gpu_metrics_name = _check_service_started(docker_client, label=GPU_METRICS_EXPORTER_CONTAINER_LABEL)
    rand_num = random.randint(0, 100000)

    if dcgm_name is None:
        dcgm_name = f'dcgm-exporter-{rand_num}'
        # start dcgm-exporter
        docker_client.containers.run(
            'bgbiao/dcgm-exporter', detach=True, runtime='nvidia', name=dcgm_name,
            labels=[DCGM_EXPORTER_CONTAINER_LABEL],
        )

    check_container_status(docker_client, dcgm_name)
    logger.info(f'Container name={dcgm_name} started.')

    if gpu_metrics_name is None:
        gpu_metrics_name = f'gpu-metrics-exporter-{rand_num}'
        dcgm_container = list_containers(docker_client, filters={'name': dcgm_name})[0]
        # start gpu-metric-exporter
        docker_client.containers.run(
            'bgbiao/gpu-metrics-exporter', detach=True, privileged=True, name=gpu_metrics_name,
            ports={'9400/tcp': port}, volumes_from=[dcgm_container.id], labels=[GPU_METRICS_EXPORTER_CONTAINER_LABEL]
        )

    check_container_status(docker_client, gpu_metrics_name)
    logger.info(f'{dcgm_name} stared')


def stop_node_exporter(
        docker_client,
):
    """Stop node exporter service.

    Args:
        docker_client (docker.client.DockerClient): Docker client instance.
    """
    _stop_service(docker_client, DCGM_EXPORTER_CONTAINER_LABEL)
    _stop_service(docker_client, GPU_METRICS_EXPORTER_CONTAINER_LABEL)


def start_fastapi_backend():
    # check if the process is running
    pid = _check_process_running('/tmp/fastapi_backend')
    if not pid:
        args = [sys.executable, '-m', 'uvicorn', 'modelci.app.main:app', '--host', SERVER_HOST, '--port',
                str(SERVER_PORT)]
        backend_process = subprocess.Popen(args, preexec_fn=os.setsid)
        # save the process pid
        with open('/tmp/fastapi_backend', 'w') as f:
            f.write(str(backend_process.pid))
    logger.info('FastAPI started.')


def stop_fastapi_backend():
    # get backend process pid
    pid = _check_process_running('/tmp/fastapi_backend')
    if pid:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        logger.info(f'FastAPI PID={pid} stopped.')


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
        get_image(docker_client, image, logger)


if __name__ == '__main__':
    start()
    # stop()
