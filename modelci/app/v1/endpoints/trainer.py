#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/15/2021
"""
from fastapi import APIRouter
from modelci.types.vo.model_vo import TrainerConfig
router = APIRouter()


@router.post('/{model_id}')
def submit(model_id: str, train_config: TrainerConfig):
    """
    Submit a training job.
    TODO(JSS): add model structure modifications to TrainerConfig
    TODO(JSS): connect to finertuner

    Args:
        model_id (str): Model ID for training.
        train_config (TrainerConfig): Training related parameters

    Returns:
        Submit status.
    """
    print(model_id)
    print(train_config)
    return {"id": 1}
