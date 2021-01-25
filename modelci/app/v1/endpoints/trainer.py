#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/15/2021
"""
from fastapi import APIRouter

router = APIRouter()


@router.post('/')
def submit(model_id: str, other_body: ...):
    """
    Submit a training job to the controller.

    Args:
        model_id (str): Model ID for training.
        other_body: Other required parameters for training.

    Returns:
        Submitted training job data class object.
    """
    raise NotImplementedError('Method `submit` not implemented')
