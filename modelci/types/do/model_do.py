# -*- coding: utf-8 -*-
"""Module for model plain object."""

from mongoengine import Document, EmbeddedDocument
from mongoengine.fields import (
    DateTimeField,
    EmbeddedDocumentField,
    EmbeddedDocumentListField,
    FileField,
    FloatField,
    IntField,
    ListField,
    StringField,
    DictField
)

from .profile_result_do import ProfileResultDO


class IOShapeDO(EmbeddedDocument):
    name = StringField()
    shape = ListField(IntField(), required=True)
    dtype = StringField(required=True)
    format = IntField(required=True)


class ModelDO(Document):
    """Model Plain Object.

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
    # Model evaluation metric
    metric = DictField(required=True)
    # Model weights
    weight = FileField()
    # Model task
    task = IntField(required=True)
    # Parent Model ID
    parent_model_id = StringField()
    # inputs standard
    inputs = EmbeddedDocumentListField(IOShapeDO)
    # outputs standard
    outputs = EmbeddedDocumentListField(IOShapeDO)
    # Profile result
    profile_result = EmbeddedDocumentField(ProfileResultDO)
    # Status enum value
    status = IntField(required=True)
    # Model Status enum value
    model_status = ListField()
    # Model provider (uploader)
    creator = StringField(required=True)
    # Creation time of this record
    create_time = DateTimeField(required=True)

    meta = {
        'indexes': [
            {'fields': ('engine', 'name', 'framework', 'version', 'task'), 'unique': True}
        ]
    }
