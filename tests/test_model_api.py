#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 10/14/2020
"""


def test_upload_model():
    # Register a model
    # TODO: use `curl` to register a model
    # > curl -X POST "http://155.69.146.35:8000/api/v1/model/?convert=true&profile=false" \
    #   -H  "accept: application/json" -H  "Content-Type: multipart/form-data" \
    #  -F "architecture=ResNet50" -F "model_input=" -F "dataset=ImageNet" -F "version=1" -F "framework=PyTorch" \
    #   -F "engine=PYTORCH" -F "task=Image_Classification"  -F "metric={\"acc\": 0.76}" \
    #   -F "inputs=[{\"name\": \"input\", \"shape\": [-1, 3, 224, 224], \"dtype\": \"TYPE_FP32\", \"format\": \"FORMAT_NCHW\"}]" \
    #   -F "outputs=[{\"name\": \"output\", \"shape\": [-1, 1000], \"dtype\": \"TYPE_FP32\"}]"
    # \
    #   "files=@${HOME}/.modelci/ResNet50/pytorch-pytorch/image_classification/1.pth"

    # Expected result
    # {
    #   "data": {
    #     "id": [
    #       "603a9de960823dadedc60763",
    #       "603a9dea60823dadedc608ee",
    #       "603a9deb60823dadedc60a79"
    #     ]
    #   },
    #   "status":true
    # }
    pass
