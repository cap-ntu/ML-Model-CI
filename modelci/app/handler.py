#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/22/2020
"""
from datetime import datetime

from fastapi import Request
from starlette.responses import JSONResponse

from modelci.app.main import app
from modelci.persistence.exceptions import DoesNotExistException, BadRequestValueException, ServiceException


@app.exception_handler(ServiceException)
async def service_exception_handler(request: Request, exc: ServiceException):
    return JSONResponse(
        status_code=500,
        content={'message': exc.message, 'status': 500, 'timestamp': datetime.now().timestamp()}
    )


@app.exception_handler(DoesNotExistException)
async def does_not_exist_exception_handler(request: Request, exc: DoesNotExistException):
    return JSONResponse(
        status_code=404,
        content={'message': exc.message, 'status': 404, 'timestamp': datetime.now().timestamp()}
    )


@app.exception_handler(BadRequestValueException)
async def bad_request_value_exception_handler(request: Request, exc: BadRequestValueException):
    return JSONResponse(
        status_code=400,
        content={'message': exc.message, 'status': 400, 'timestamp': datetime.now().timestamp()}
    )
