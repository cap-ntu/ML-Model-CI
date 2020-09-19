#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 9/19/2020
"""
from .onnx_client import CVONNXClient
from .tfs_client import CVTFSClient
from .torch_client import CVTorchClient
from .trt_client import CVTRTClient

__all__ = ['CVONNXClient', 'CVTFSClient', 'CVTRTClient', 'CVTorchClient']
