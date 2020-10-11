#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 10/2/2020
"""
from modelci.app import (start as app_start, stop as app_stop)
from modelci.utils import Logger
from modelci.utils.docker_container_manager import DockerContainerManager

logger = Logger(__name__, welcome=False)

backend_process = None

container_conn = DockerContainerManager()


def start():
    """Start the ModelCI service."""
    if container_conn.start():
        container_conn.connect()
    app_start()


def stop():
    """Stop the ModelCI service"""
    container_conn.stop()
    app_stop()


if __name__ == '__main__':
    start()
    # stop()
