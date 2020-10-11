#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 10/8/2020
"""
import random
import subprocess
import time
from pathlib import Path

import docker
from pymongo import MongoClient

from modelci.config import MONGO_HOST, MONGO_PORT, NODE_EXPORTER_PORT, MONGO_USERNAME, MONGO_PASSWORD, MONGO_DB, \
    CADVISOR_PORT
from modelci.utils import Logger
from modelci.utils.docker_api_utils import list_containers, get_image, check_container_status

MODELCI_DOCKER_LABEL = 'modelci.container.label'

MODELCI_DOCKER_PORT_LABELS = {
    'mongo': 'modelci.mongo.port',
    'cadvisor': 'modelci.cadvisor.port',
    'dcgm_node_exporter': 'modelci.dcgm-node-exporter.port',
    'gpu_metrics_node_exporter': 'modelci.gpu-metrics-node-exporter.port',
    'backend': 'modelci.backend.port',
}


class DockerContainerManager(object):
    def __init__(
            self,
            cluster_name='default-cluster',
            mongo_ip=MONGO_HOST,
            mongo_port=MONGO_PORT,
            cadvisor_port=CADVISOR_PORT,
            node_exporter_port=NODE_EXPORTER_PORT,
            docker_network='modelci_network',
            extra_container_kwargs=None,
            enable_gpu=False,
    ):
        self.cluster_name = cluster_name
        self.mongo_ip = mongo_ip
        self.mongo_port = mongo_port
        self.cadvisor_port = cadvisor_port
        self.node_exporter_port = node_exporter_port
        self.docker_network = docker_network
        self.enable_gpu = enable_gpu

        self.docker_client = docker.from_env()
        if extra_container_kwargs is None:
            self.extra_container_kwargs = {}
        else:
            self.extra_container_kwargs = extra_container_kwargs.copy()

        # Merge Clipper-specific labels with any user-provided labels
        if 'labels' in self.extra_container_kwargs:
            self.common_labels = self.extra_container_kwargs.pop('labels')
            self.common_labels.update({MODELCI_DOCKER_LABEL: self.cluster_name})
        else:
            self.common_labels = {MODELCI_DOCKER_LABEL: self.cluster_name}

        container_kwargs = {'detach': True}

        self.extra_container_kwargs.update(container_kwargs)

        # create logger
        self.logger = Logger('ml-modelci Docker Container Manager', welcome=False)

    def start(self):
        """Start the ModelCI service."""

        containers_in_cluster = list_containers(
            docker_client=self.docker_client,
            filters={'label': [f'{MODELCI_DOCKER_LABEL}={self.cluster_name}']}
        )

        if containers_in_cluster == 4:
            self.logger.error(f'Cluster {self.cluster_name} cannot be started because it already exists.')
            return False

        # download all required docker images
        self._download_serving_containers()

        # obtain which containers has started
        all_labels = dict()
        for container in containers_in_cluster:
            all_labels.update(container.labels)
            if container.attrs['State']['Status'] != 'running':
                # try start stopped container
                self.logger.warning(f'Service already exist, found container name={container.name}.')
                container.start()
                self.logger.info('Service started.')
            else:
                self.logger.warning(f'Service with container name={container.name} already started.')

        if not MODELCI_DOCKER_PORT_LABELS['mongo'] in all_labels:
            self._start_mongo_db()

        if not MODELCI_DOCKER_PORT_LABELS['cadvisor'] in all_labels:
            self._start_cadvisor()

        if not MODELCI_DOCKER_PORT_LABELS['dcgm_node_exporter'] in all_labels:
            self._start_dcgm_node_exporter()

        if not MODELCI_DOCKER_PORT_LABELS['gpu_metrics_node_exporter'] in all_labels:
            self._start_gpu_metrics_node_exporter()

        return self.connect()

    def connect(self):
        """Use the cluster name to update ports. Because they might not match as in
        start_clipper the ports might be changed.
        """
        containers = list_containers(
            docker_client=self.docker_client,
            filters={'label': [f'{MODELCI_DOCKER_LABEL}={self.cluster_name}']})
        all_labels = dict()
        for container in containers:
            all_labels.update(container.labels)

        self.mongo_port = all_labels[MODELCI_DOCKER_PORT_LABELS['mongo']]
        self.cadvisor_port = all_labels[MODELCI_DOCKER_PORT_LABELS['cadvisor']]
        self.node_exporter_port = all_labels[MODELCI_DOCKER_PORT_LABELS['gpu_metrics_node_exporter']]

        return True

    def stop(self):
        containers = list_containers(
            self.docker_client, filters={'label': [f'{MODELCI_DOCKER_LABEL}={self.cluster_name}'], 'status': 'running'}
        )
        for container in containers:
            container.stop()
            self.logger.info(f'Container name={container.name} stopped.')

    def _download_serving_containers(self):
        images = [
            'mlmodelci/pytorch-serving:latest',
            'mlmodelci/onnx-serving:latest',
            'tensorflow/serving:2.1.0',
            'mongo:latest',
            'google/cadvisor:latest',
            'bgbiao/dcgm-exporter:latest',
            'bgbiao/gpu-metrics-exporter:latest',
        ]

        if self.enable_gpu:
            images.extend([
                'mlmodelci/pytorch-serving:latest-gpu',
                'mlmodelci/onnx-serving:latest-gpu',
                'tensorflow/serving:2.1.0-gpu',
                'nvcr.io/nvidia/tensorrtserver:19.10-py3',
            ])

        for image in images:
            get_image(self.docker_client, image, self.logger)

    def _start_mongo_db(self):
        """Start Mongo DB service.
            From https://stackoverflow.com/a/53522699/13173608.
        """
        mongo_name = f'mongo-{random.randint(0, 100000)}'

        self.docker_client.containers.run(
            'mongo', ports={'27017/tcp': self.mongo_port}, name=mongo_name,
            environment={
                'MONGO_INITDB_USERNAME': MONGO_USERNAME,
                'MONGO_INITDB_PASSWORD': MONGO_PASSWORD,
                'MONGO_INITDB_DATABASE': MONGO_DB,
            },
            labels={**self.common_labels, MODELCI_DOCKER_PORT_LABELS['mongo']: str(self.mongo_port)},
            **self.extra_container_kwargs
        )

        time.sleep(1)
        try:
            # create MongoDB user
            client = MongoClient(f'{MONGO_HOST}:{MONGO_PORT}')
            kwargs = {'pwd': MONGO_PASSWORD, 'roles': [{'role': 'readWrite', 'db': MONGO_DB}]}
            getattr(client, MONGO_DB).command("createUser", MONGO_USERNAME, **kwargs)
        except Exception as e:
            self.logger.error(f'Exception during starting MongoDB: {e}')
            container = list_containers(self.docker_client, filters={'name': mongo_name})[0]
            container.kill()
            container.remove()
            return

        check_container_status(self.docker_client, name=mongo_name)
        self.logger.info(f'Container name={mongo_name} stared')

    def _start_cadvisor(self):
        """Start cAdvisor service."""
        cadvisor_name = f'cadvisor-{random.randint(0, 100000)}'

        volumes = {
            '/': {'bind': '/rootfs', 'mode': 'ro'},
            '/var/run': {'bind': '/var/run', 'mode': 'rw'},
            '/sys': {'bind': '/sys', 'mode': 'ro'},
            '/var/lib/docker': {'bind': '/var/lib/docker', 'mode': 'ro'},
        }

        extra_container_kwargs = self.extra_container_kwargs.copy()

        if self.enable_gpu:
            # find libnvidia-ml.so.1
            cache_file = Path('/tmp/libnvidia-ml.cache')
            if cache_file.exists():
                with open(cache_file) as f:
                    libnvidia_ml_path = f.read().strip()
            else:
                args1 = ('locate', 'libnvidia-ml.so.1')
                args2 = ('grep', '-v', 'lib32')
                args3 = ('head', '-1')
                locate = subprocess.Popen(args1, stdout=subprocess.PIPE)
                grep = subprocess.Popen(args2, stdin=locate.stdout, stdout=subprocess.PIPE)
                locate.wait()
                grep.wait()
                libnvidia_ml_path = subprocess.check_output(
                    args3, stdin=grep.stdout, universal_newlines=True, text=True
                ).strip()

                # save to cache
                with open(cache_file, 'w') as f:
                    f.write(libnvidia_ml_path)

            volumes.update({libnvidia_ml_path: {'bind': libnvidia_ml_path}})
            extra_container_kwargs.update({'environment': {'LD_LIBRARY_PATH': str(Path(libnvidia_ml_path).parent)}})

        self.docker_client.containers.run(
            'google/cadvisor:latest', name=cadvisor_name, ports={'8080/tcp': self.cadvisor_port},
            privileged=True, volumes=volumes,
            labels={**self.common_labels, MODELCI_DOCKER_PORT_LABELS['cadvisor']: str(self.cadvisor_port)},
            **extra_container_kwargs
        )

        check_container_status(self.docker_client, name=cadvisor_name)
        self.logger.info(f'Container name={cadvisor_name} started.')

    def _start_dcgm_node_exporter(self):
        """Start node exporter service."""
        rand_num = random.randint(0, 100000)

        dcgm_name = f'dcgm-exporter-{rand_num}'
        # start dcgm-exporter
        self.docker_client.containers.run(
            'bgbiao/dcgm-exporter', runtime='nvidia', name=dcgm_name,
            labels={**self.common_labels, MODELCI_DOCKER_PORT_LABELS['dcgm_node_exporter']: '-1'},
            **self.extra_container_kwargs
        )

        check_container_status(self.docker_client, dcgm_name)
        self.logger.info(f'Container name={dcgm_name} started.')

    def _start_gpu_metrics_node_exporter(self):
        rand_num = random.randint(0, 100000)
        gpu_metrics_name = f'gpu-metrics-exporter-{rand_num}'
        dcgm_container = list_containers(
            self.docker_client, filters={'label': [MODELCI_DOCKER_PORT_LABELS['dcgm_node_exporter']]}
        )[0]
        # start gpu-metric-exporter
        self.docker_client.containers.run(
            'bgbiao/gpu-metrics-exporter', privileged=True, name=gpu_metrics_name,
            ports={'9400/tcp': self.node_exporter_port}, volumes_from=[dcgm_container.id],
            labels={
                **self.common_labels,
                MODELCI_DOCKER_PORT_LABELS['gpu_metrics_node_exporter']: str(self.node_exporter_port)
            },
            **self.extra_container_kwargs
        )

        check_container_status(self.docker_client, gpu_metrics_name)
        self.logger.info(f'{gpu_metrics_name} stared')
