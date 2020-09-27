#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 9/19/2020
"""
import platform
import subprocess
import sys
import tarfile
from distutils.core import setup
from pathlib import Path

import requests
from setuptools import find_packages

################################################################################
# Check Platform and Python Version
################################################################################

if sys.version_info < (3,):
    print('Python 2 has reached end-of-life and is no longer supported by PyTorch.')
    sys.exit(-1)
if sys.platform != 'linux':
    TRITON_CLIENT_INSTALL = False
else:
    TRITON_CLIENT_INSTALL = True

python_min_version = (3, 7, 0)
python_min_version_str = '.'.join((str(num) for num in python_min_version))
python_max_version = (3, 9, 0)
python_max_version_str = '.'.join((str(num) for num in python_max_version))
if sys.version_info < python_min_version or sys.version_info >= python_max_version:
    print(
        f'You are using Python {platform.python_version()}. Python >={python_min_version_str},<{python_max_version_str}'
        f' is required.'
    )
    sys.exit(-1)

################################################################################
# Download Required non-pip Packages
################################################################################

TRITON_CLIENT_VERSION = '1.8.0'

# Get Ubuntu version
check_ubuntu_version_args = ['/usr/bin/lsb_release', '-sr']
stdout = subprocess.check_output(check_ubuntu_version_args, universal_newlines=True, shell=False)
try:
    UBUNTU_VERSION = int(stdout.strip().replace('.', ''))
except ValueError:
    print('You are not using UBUNTU, Triton Client is not available.')
    TRITON_CLIENT_INSTALL = False


def install_triton_client():
    filename = f'v{TRITON_CLIENT_VERSION}_ubuntu{UBUNTU_VERSION}.clients.tar.gz'
    save_name = Path.home() / 'tmp/tensorrtserver/tritonis.client.tar.gz'
    save_name.parent.mkdir(parents=True, exist_ok=True)

    url = f'https://github.com/triton-inference-server/server/releases/download/v{TRITON_CLIENT_VERSION}/{filename}'
    download = True
    if save_name.exists():
        # skip download
        try:
            tar_file = tarfile.open(save_name, mode='r')
            tar_file.extractall()
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
        tar_file.extractall()
        tar_file.close()

    package_path = save_name.parent / f'python/tensorrtserver-{TRITON_CLIENT_VERSION}-py2.py3-none-linux_x86_64.whl'

    return package_path


# parse required packages
with open('requirements.txt') as f:
    install_requires = list()
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
    name='modelci',
    version='1.0.0',
    description='A complete platform for managing, converting, profiling, and deploying models as cloud services (MLaaS)',
    author='NTU CAP',
    author_email='huaizhen001@e.ntu.edu.sg',
    url='https://github.com/cap-ntu/ML-Model-CI',
    install_requires=install_requires,
    packages=find_packages(),
    python_requires='>=3.7',
)
