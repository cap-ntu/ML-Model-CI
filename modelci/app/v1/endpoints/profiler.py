#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Xing Di
Date: 2021/1/15

"""
from fastapi import APIRouter, HTTPException
from modelci.persistence.service_ import exists_by_id, profile_model

router = APIRouter()


@router.get('/{model_id}', status_code=201)
def profile(model_id: str, device: str='cuda', batch_size: int=1):
    if not exists_by_id(model_id):
        raise HTTPException(
            status_code=404,
            detail=f'Model ID {model_id} does not exist. You may change the ID',
        )
    profile_result = profile_model(model_id, device, batch_size)
    return profile_result
