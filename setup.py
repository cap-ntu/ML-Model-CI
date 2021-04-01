#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 9/19/2020

"""
import os
import platform
import sys
import tarfile
import warnings
from distutils.core import setup
from pathlib import Path

import distro
import requests
from setuptools import find_packages

################################################################################
# Check Python Version
################################################################################

if sys.version_info < (3,):
    sys.exit('Python 2 has reached end-of-life and is no longer supported by PyTorch.')

python_min_version = (3, 7, 0)
python_min_version_str = '.'.join((str(num) for num in python_min_version))
python_max_version = (3, 9, 0)
python_max_version_str = '.'.join((str(num) for num in python_max_version))
if sys.version_info < python_min_version or sys.version_info >= python_max_version:
    sys.exit(
        f'You are using Python {platform.python_version()}. Python >={python_min_version_str},<{python_max_version_str}'
        f' is required.'
    )

################################################################################
# Check Platform
################################################################################

TRITON_CLIENT_VERSION = '1.8.0'
# Get Ubuntu version
system_name, system_version, _ = distro.linux_distribution()
if sys.platform == 'linux' and system_name == 'ubuntu':
    TRITON_CLIENT_INSTALL = True
    UBUNTU_VERSION = system_version
else:
    TRITON_CLIENT_INSTALL = False
    warnings.warn('You are not using UBUNTU, Triton Client is not available.')


def install_triton_client():
    filename = f'v{TRITON_CLIENT_VERSION}_ubuntu{UBUNTU_VERSION}.clients.tar.gz'
    save_name = Path.home() / 'tmp/tensorrtserver/tritonis.client.tar.gz'
    save_name.parent.mkdir(parents=True, exist_ok=True)

    url = f'https://github.com/triton-inference-server/server/releases/download/v{TRITON_CLIENT_VERSION}/{filename}'
    wheel_file = f'tensorrtserver-{TRITON_CLIENT_VERSION}-py2.py3-none-linux_x86_64.whl'
    download = True
    if save_name.exists():
        # skip download
        try:
            tar_file = tarfile.open(save_name, mode='r')
            tar_file.extract(f'python/{wheel_file}', path=save_name.parent)
            tar_file.close()
            download = False
        except Exception as e:
            print(f'When trying to extract {save_name}, an exception raised: {e}', file=sys.stderr)
            print(f'Re-download from {url}')
    if download:
        response = requests.get(url, allow_redirects=True)
        with open(save_name, 'wb') as f:
            f.write(response.content)
        response.close()
        tar_file = tarfile.open(save_name, mode='r')
        tar_file.extract(f'python/{wheel_file}', path=save_name.parent)
        tar_file.close()

    package_path = save_name.parent / f'python/{wheel_file}'

    return package_path


# get torch ,torchvision and tensorflow version
# reference: https://github.com/openvinotoolkit/nncf/blob/develop/setup.py
# reference: https://www.tensorflow.org/install/source#gpu
PRECOMPILED_TENSORFLOW_PAIRS = {
    "cu110": {
        "tensorflow": "tensorflow-gpu==2.4.0",
        "tensorflow-serving-api": "tensorflow-serving-api-gpu==2.4.0"
    },
    "cu102": {
        "tensorflow": "tensorflow-gpu==2.3.0",
        "tensorflow-serving-api": "tensorflow-serving-api-gpu==2.3.0"
    },
    "cu101": {
        "tensorflow": "tensorflow-gpu==2.3.0",
        "tensorflow-serving-api": "tensorflow-serving-api-gpu==2.3.0"
    },
    "cpu": {
        "tensorflow": "tensorflow==2.3.0",
        "tensorflow-serving-api": "tensorflow-serving-api==2.3.0"
    }
}

CUDA_VERSION = "cpu"
PYTHON_VER = f'{sys.version_info[0]}{sys.version_info[1]}'
torch_cuda_version = "cpu"
tensorflow_cuda_version = "cpu"


if "CUDA_HOME" in os.environ:
    cuda_version_file = os.path.join(os.environ["CUDA_HOME"], "version.txt")
    if os.path.exists(cuda_version_file):
        with open(cuda_version_file) as f:
            CUDA_VERSION = "cu".join(f.readline().strip().split(" ")[-1].split(".")[:2])
            if CUDA_VERSION not in ["cpu", "cu92", "cu101", "cu102"]:
                warnings.warn(
                    f"There is no pre-complied pytorch 1.5.0 with CUDA {CUDA_VERSION}, "
                    f"and you might need to install pytorch 1.5.0 with CUDA {CUDA_VERSION} from source."
                )
            else:
                torch_cuda_version = CUDA_VERSION

            if CUDA_VERSION not in PRECOMPILED_TENSORFLOW_PAIRS:
                warnings.warn(
                    f"There is no pre-complied tensorflow-gpu >=2.1.0 with CUDA {CUDA_VERSION}, "
                    f"and you might need to install tensorflow-gpu >=2.1.0 with CUDA {CUDA_VERSION} from source."
                )
            else:
                tensorflow_cuda_version = CUDA_VERSION

TORCH_INSTALL_URL = f'https://download.pytorch.org/whl/{torch_cuda_version}/torch-1.5.0' \
                    f'{"%2B"+torch_cuda_version if torch_cuda_version!="cu102" else ""}' \
                    f'-cp{PYTHON_VER}-cp{PYTHON_VER+"m" if PYTHON_VER!="38" else PYTHON_VER}-linux_x86_64.whl '
TORCHVISION_INSTALL_URL = f'https://download.pytorch.org/whl/{torch_cuda_version}/torchvision-0.6.0' \
                          f'{"%2B"+torch_cuda_version if torch_cuda_version!="cu102" else ""}' \
                          f'-cp{PYTHON_VER}-cp{PYTHON_VER+"m" if PYTHON_VER!="38" else PYTHON_VER}-linux_x86_64.whl'

TENSORFLOW_REQ = PRECOMPILED_TENSORFLOW_PAIRS[tensorflow_cuda_version]["tensorflow"]
TFS_REQ = PRECOMPILED_TENSORFLOW_PAIRS[tensorflow_cuda_version]["tensorflow-serving-api"]

# parse required packages
install_requires = list()
install_requires.append(f"torch @ {TORCH_INSTALL_URL}")
install_requires.append(f"torchvision @ {TORCHVISION_INSTALL_URL}")
install_requires.append(TENSORFLOW_REQ)
install_requires.append(TFS_REQ)
install_requires.append("torchviz==0.0.1")
install_requires.append("pytorch-lightning==1.1.4")

with open('requirements.txt') as f:
    for line in f.readlines():
        install_requires.append(line.strip())

if TRITON_CLIENT_INSTALL:
    # add Triton client package
    triton_client_package = install_triton_client()
    install_requires.append(f'tensorrtserver@ file://{triton_client_package}')

################################################################################
# Pip Install Packages
################################################################################

setup(
    install_requires=install_requires,
    packages=find_packages(),
    DEPENDENCY_LINKS=[TORCH_INSTALL_URL, TORCHVISION_INSTALL_URL],
    entry_points='''
        [console_scripts]
        modelci=modelci.cli:cli
    '''
)
