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

from modelci.app.config import SERVER_HOST, SERVER_PORT
from modelci.utils import Logger
from modelci.utils.misc import check_process_running

logger = Logger('modelci backend', welcome=False)


def start():
    """Run a ModelCI backend server with Uvicorn."""
    # check if the process is running
    pid = check_process_running(SERVER_PORT)
    if not pid:
        args = [sys.executable, '-m', 'uvicorn', 'modelci.app.main:app', '--port',
                str(SERVER_PORT), '--host', '0.0.0.0']
        if SERVER_HOST != 'localhost':
            args += ['--host', SERVER_HOST]
        backend_process = subprocess.Popen(args, preexec_fn=os.setsid)
        pid = backend_process.pid
        logger.info(f'Uvicorn server listening on {SERVER_PORT}')
    else:
        logger.warning(f'Unable to started server. A process with pid={pid} is already listening on {SERVER_PORT}. '
                       'Please check if your Uvicorn server has started.')


def stop():
    """Stop the ModelCI backend server."""
    # get backend process pid
    pid = check_process_running(SERVER_PORT)
    if pid:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        logger.info(f'The Uvicorn server with pid={pid} stopped.')
    else:
        logger.warning(f'No process is listening on {SERVER_PORT}')
