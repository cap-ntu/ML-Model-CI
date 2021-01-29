#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/19/2020
"""

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from modelci.app import config
from modelci.app.experimental.api import api_router as api_rounter_exp
from modelci.app.v1.api import api_router

app = FastAPI(title=config.PROJECT_NAME, openapi_url="/api/v1/openapi.json")

# CORS
origins = []

# Set all CORS enabled origins
if config.BACKEND_CORS_ORIGINS:
    origins_raw = config.BACKEND_CORS_ORIGINS.split(",")
    for origin in origins_raw:
        use_origin = origin.strip()
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

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
