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
from pathlib import Path

from setuptools import setup, find_packages

if sys.version_info < (3,):
    print('Python 2 has reached end-of-life and is no longer supported by PyTorch.')
    sys.exit(-1)
if sys.platform != 'linux':
    print('Only support Linux system.')
    sys.exit(-1)

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

TRITON_CLIENT_INSTALL = True
TRITON_CLIENT_VERSION = '1.8.0'

process = subprocess.Popen(['lsb_release', '-sr'], stdout=subprocess.PIPE, universal_newlines=True)
stdout, _ = process.communicate()
try:
    UBUNTU_VERSION = int(stdout.strip().replace('.', ''))
except ValueError:
    print('You are not using UBUNTU, Triton Client is not available.')
    TRITON_CLIENT_INSTALL = False


def install_triton_client():
    import requests

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

    subprocess.call([sys.executable, '-m', 'pip', 'install', package_path])


# parse required packages
with open('requirements.txt') as f:
    install_requires = list()
    for line in f.readlines():
        install_requires.append(line.strip())

setup(
    name='modelci',
    version='1.0.0',
    description='A complete platform for managing, converting, profiling, and deploying models as cloud services (MLaaS)',
    author='Yuanming Li',
    author_email='yli056@e.ntu.edu.sg',
    url='https://github.com/cap-ntu/ML-Model-CI',
    install_requires=install_requires,
    packages=find_packages(),
    python_requires='>=3.7',
)

install_triton_client()
