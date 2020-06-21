#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/20/2020
"""
from fastapi import APIRouter

from modelci.types.vo.model_ao import ModelOut

router = APIRouter()


@router.get('/{id}', response_model=ModelOut)
def get_model(*, id: int):
    pass
