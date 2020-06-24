#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/20/2020
"""
from fastapi import APIRouter

from modelci.app.v1.endpoints import model

api_router = APIRouter()
api_router.include_router(model.router, prefix='/model', tags=['model'])
