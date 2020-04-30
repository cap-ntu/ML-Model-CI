# -*- coding: utf-8 -*-
"""Module for model plain object."""

from mongoengine import Document, EmbeddedDocument
from mongoengine.fields import *

from .profile_result_po import ProfileResultPO


class IOShapePO(EmbeddedDocument):
    name = StringField()
    shape = ListField(IntField(), required=True)
    dtype = StringField(required=True)
    format = IntField(required=True)


class ModelPO(Document):
    """
    Model Plain Object.

    The primary key of the model plain object is (engine, name, version) pair.
    """

    # Model name
    name = StringField(required=True)
    # Supported engine enum (aka framework, e.g.: TensorFlow (0) or PyTorch (1))
    framework = IntField(required=True)
    # ONNX or TensorRT
    engine = IntField(required=True)
    # Version of the model. e.g.: `1`
    version = IntField(required=True)
    # Dataset
    dataset = StringField(required=True)
    # Model accuracy
    accuracy = FloatField(required=True)
    # Model weights
    weight = FileField()
    # Model task
    task = StringField(required=True)
    # inputs standard
    inputs = EmbeddedDocumentListField(IOShapePO)
    # outputs standard
    outputs = EmbeddedDocumentListField(IOShapePO)
    # Profile result
    profile_result = EmbeddedDocumentField(ProfileResultPO)
    # Status enum value
    status = IntField(required=True)
    # Creation time of this record
    create_time = DateTimeField(required=True)

    meta = {
        'indexes': [
            {'fields': ('engine', 'name', 'framework', 'version'), 'unique': True}
        ]
    }
