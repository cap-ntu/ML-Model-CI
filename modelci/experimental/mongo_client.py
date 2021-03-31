#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/1/2021
"""
from typing import Optional

import pymongo

from modelci.config import db_settings


class MongoClient(pymongo.MongoClient):
    def __init__(
            self,
            host: str = db_settings.mongo_host,
            port: str = db_settings.mongo_port,
            document_class: type = dict,
            tz_aware: bool = True,
            connect: bool = None,
            type_registry=None,
            username: str = db_settings.mongo_username,
            password: Optional[str] = db_settings.mongo_password.get_secret_value(),
            authSource: str = db_settings.mongo_auth_source,
            authMechanism: str = db_settings.auth_mechanism,
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
