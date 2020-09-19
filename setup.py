#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 9/19/2020
"""
from setuptools import setup, find_packages

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
