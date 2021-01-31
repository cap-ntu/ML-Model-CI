#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/20/2020
"""
from fastapi import APIRouter

from modelci.app.experimental.endpoints import cv_tuner
from modelci.app.experimental.endpoints import model_structure

api_router = APIRouter()
api_router.include_router(cv_tuner.router, prefix='/cv-tuner', tags=['[*exp] cv-tuner'])
api_router.include_router(model_structure.router, prefix='/structure', tags=['[*exp] structure'])
