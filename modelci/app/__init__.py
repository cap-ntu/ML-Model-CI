#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/19/2020
"""
import multiprocessing as mp
import os
import signal
from pathlib import Path

from modelci.config import app_settings
from modelci.utils import Logger
from modelci.utils.misc import check_process_running

logger = Logger('modelci backend', welcome=False)
default_log_file = Path.home() / 'tmp/modelci.log'
default_log_file.parent.mkdir(exist_ok=True)


def start():
    """Run a ModelCI backend server with Uvicorn."""
    from modelci.app.main import _app_start_detach

    # check if the process is running
    pid = check_process_running(app_settings.server_port)
    if not pid:
        mp.set_start_method('spawn')
        backend_process = mp.Process(target=_app_start_detach, args=(default_log_file,))
        backend_process.start()

        logger.info(f'Uvicorn server listening on {app_settings.server_url}, check full log at {default_log_file}')
    else:
        logger.warning(f'Unable to started server. A process with pid={pid} is already listening on '
                       f'port {app_settings.server_port}. '
                       'Please check if your Uvicorn server has started.')


def stop():
    """Stop the ModelCI backend server."""
    # get backend process pid
    pid = check_process_running(app_settings.server_port)
    if pid:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        logger.info(f'The Uvicorn server with pid={pid} stopped.')
    else:
        logger.warning(f'No process is listening on port {app_settings.server_port}')
