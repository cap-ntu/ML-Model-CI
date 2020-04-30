from mongoengine import EmbeddedDocument
from mongoengine.fields import *

from .dynamic_profile_result_po import DynamicProfileResultPO
from .static_profile_result_po import StaticProfileResultPO


class ProfileResultPO(EmbeddedDocument):
    """
    Profiling result plain object.
    """
    # Static profile result
    static_profile_result = EmbeddedDocumentField(StaticProfileResultPO)
    # Dynamic profile result
    dynamic_profile_results = ListField(EmbeddedDocumentField(DynamicProfileResultPO))
