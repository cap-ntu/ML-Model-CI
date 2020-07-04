from mongoengine import EmbeddedDocument
from mongoengine.fields import IntField, LongField


class StaticProfileResultDO(EmbeddedDocument):
    """
    Static profiling result plain object
    """

    # Number of parameters of this model
    parameters = IntField(required=True)
    # Floating point operations
    flops = LongField(required=True)
    # Memory consumption in Byte in order to load this model into GPU or CPU
    memory = LongField(required=True)
    # Memory read in Byte
    mread = LongField(required=True)
    # Memory write in Byte
    mwrite = LongField(required=True)
    # Memory readwrite in Byte
    mrw = LongField(required=True)
