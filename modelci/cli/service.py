#  Copyright (c) NTU_CAP 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import click

from modelci.app import (start as app_start, stop as app_stop)
from modelci.utils import Logger
from modelci.utils.docker_container_manager import DockerContainerManager

logger = Logger(__name__, welcome=False)

@click.group()
def service():
    pass


@service.command("init")
@click.option('--gpu', default=False, type=click.BOOL, is_flag=True)
def start_leader_server(gpu=False):
    """start the system on a leader server in a cluster.
    initialize necessary services such as database and monitor

    Args:
        gpu (bool, optional): [description]. Defaults to False.
    """
    # todo: lazy docker start
    container_conn = DockerContainerManager(enable_gpu=gpu)
    if not container_conn.start():
        container_conn.connect()
    app_start()

@service.command("stop")
def stop_leader_server():
    """shutdown all jobs and save a screenshot, then stop the leader server
    it will broadcast a message to all follower workers
    """
    container_conn = DockerContainerManager()
    container_conn.stop()
    app_stop()

@service.command("clean")
def remove_services():
    """stop all services and remove downloaded docker images
    """
    container_conn = DockerContainerManager()
    container_conn.stop()
    app_stop()
    container_conn.remove_all()


@service.command("connect")
@click.argument("ip_address")
def connect_leader_server(ip_address="localhost"):
    """connect to the leader server to team use

    Args:
        ip_address (str, optional): [description]. Defaults to "localhost".

    Raises:
        NotImplementedError: [description]
    """

    raise NotImplementedError

@service.command("disconnect")
def disconnect_leader_server():
    """disconnect the follower worker from the leader server

    Raises:
        NotImplementedError: [description]
    """
    raise NotImplementedError


