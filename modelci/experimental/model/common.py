#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/1/2021

Common Pydantic Class
"""
from bson import ObjectId


class ObjectIdStr(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, str):
            v = ObjectId(v)
        if not isinstance(v, ObjectId):
            raise ValueError("Not a valid ObjectId")
        return str(v)
