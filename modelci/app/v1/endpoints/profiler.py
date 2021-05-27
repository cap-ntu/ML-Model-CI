#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Xing Di
Date: 2021/1/15

"""
from fastapi import APIRouter, HTTPException
from modelci.persistence.service_ import exists_by_id, profile_model

router = APIRouter()


@router.get('/', status_code=201)
def profile(id: str, device: str='cuda', batch_size: int=1):
    if not exists_by_id(id):
        raise HTTPException(
            status_code=404,
            detail=f'Model ID {id} does not exist. You may change the ID',
        )
    profile_result = profile_model(id, device, batch_size)
    return profile_result
