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

import collections
import logging
import os
import re
import socket
import subprocess
from typing import _GenericAlias, _SpecialForm, Any  # noqa


def json_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = json_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def remove_dict_null(d: dict):
    """Remove `None` value in dictionary."""
    return {k: v for k, v in d.items() if v is not None}


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except socket.error:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def get_device(device: str):
    """Get device (cuda and device order) from device name string.

    Args:
        device: Device name string.

    Returns:
        Tuple[bool, Optional[int]]: A tuple containing flag for CUDA device and CUDA device order. If the CUDA device
            flag is `False`, the CUDA device order is `None`.
    """
    # obtain device
    device_num = None
    if device == 'cpu':
        cuda = False
    else:
        # match something like cuda, cuda:0, cuda:1
        matched = re.match(r'^cuda(?::([0-9]+))?$', device)
        if matched is None:  # load with CPU
            logging.warning('Wrong device specification, using `cpu`.')
            cuda = False
        else:  # load with CUDA
            cuda = True
            device_num = int(matched.groups()[0])
            if device_num is None:
                device_num = 0

    return cuda, device_num


def check_process_running(port: int):
    args = ['lsof', '-t', f'-i:{port}']
    try:
        pid = int(subprocess.check_output(args, universal_newlines=True, text=True, stderr=subprocess.DEVNULL))
    except subprocess.CalledProcessError:
        # process not found
        pid = None
    return pid


def make_dir(dir_path):
    if os.path.exists(dir_path):
        pass
    else:
        os.mkdir(dir_path)


def isgeneric(obj: object):
    """Return true if the object is a generic type."""
    return isinstance(obj, _GenericAlias) or isinstance(obj, _SpecialForm) and obj is not Any
