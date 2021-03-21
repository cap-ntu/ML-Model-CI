#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/19/2020
"""
import os
import sys
from typing import Optional

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from modelci import config
from modelci.app.experimental.api import api_router as api_rounter_exp
from modelci.app.v1.api import api_router

settings = config.AppSettings()
app = FastAPI(title=settings.project_name, openapi_url="/api/v1/openapi.json")

# CORS
origins = []

# Set all CORS enabled origins
if settings.backend_cors_origins:
    origins_raw = settings.backend_cors_origins.split(",")
    for origin in origins_raw:
        use_origin = origin.strip().replace('"', '')
        origins.append(use_origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=config.API_V1_STR)
app.include_router(api_rounter_exp, prefix=config.API_EXP_STR)


def _app_start_detach(output_file: Optional[str] = None):
    """Start FastAPI as a detached process.

    This is a double fork approach.

    Reference:
        https://stackoverflow.com/a/49123627
    """

    if os.fork() != 0:  # do a double fork,
        return

    import uvicorn

    if output_file:
        # redirect stdout, stderr to a file
        output_file = open(output_file, 'a')
        sys.stdout = output_file
        sys.stderr = output_file
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
    if output_file:
        output_file.close()


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
