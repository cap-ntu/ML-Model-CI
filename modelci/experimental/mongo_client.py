#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/1/2021
"""

import pymongo

from modelci.config import MONGO_HOST, MONGO_PORT, MONGO_USERNAME, MONGO_PASSWORD, MONGO_AUTH_SOURCE


class MongoClient(pymongo.MongoClient):
    def __init__(
            self,
            host: str = MONGO_HOST,
            port: str = MONGO_PORT,
            document_class: type = dict,
            tz_aware: bool = True,
            connect: bool = None,
            type_registry=None,
            username: str = MONGO_USERNAME,
            password: str = MONGO_PASSWORD,
            authSource: str = MONGO_AUTH_SOURCE,
            authMechanism: str = 'SCRAM-SHA-256',
            **kwargs
    ):
        """
        MongoDB Client wrapper with defined configuration.

        Use this class just as `pymongo.MongoClient`. We inject some database related configuration into such
        MongoDB Client wrapper, for ease of config management.
        """
        super().__init__(
            host=host,
            port=port,
            document_class=document_class,
            tz_aware=tz_aware,
            connect=connect,
            type_registry=type_registry,
            username=username,
            password=password,
            authSource=authSource,
            authMechanism=authMechanism,
            **kwargs,
        )
