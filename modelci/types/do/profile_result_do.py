from mongoengine import EmbeddedDocument
from mongoengine.fields import EmbeddedDocumentField, ListField

from .dynamic_profile_result_do import DynamicProfileResultDO
from .static_profile_result_do import StaticProfileResultDO


class ProfileResultDO(EmbeddedDocument):
    """
    Profiling result plain object.
    """
    # Static profile result
    static_profile_result = EmbeddedDocumentField(StaticProfileResultDO)
    # Dynamic profile result
    dynamic_profile_results = ListField(EmbeddedDocumentField(DynamicProfileResultDO))
