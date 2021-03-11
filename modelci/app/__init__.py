#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/19/2020
"""
import os
import signal
import subprocess
import sys

from modelci.config import app_settings
from modelci.utils import Logger
from modelci.utils.misc import check_process_running

logger = Logger('modelci backend', welcome=False)


def start():
    """Run a ModelCI backend server with Uvicorn."""
    # check if the process is running
    pid = check_process_running(app_settings.server_port)
    if not pid:
        args = [sys.executable, '-m', 'uvicorn', 'modelci.app.main:app', '--port',
                str(app_settings.server_port), '--host', '0.0.0.0']
        if app_settings.server_host != 'localhost':
            args += ['--host', app_settings.server_port]
        backend_process = subprocess.Popen(args, preexec_fn=os.setsid)
        pid = backend_process.pid
        logger.info(f'Uvicorn server listening on {app_settings.server_url}')
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
