from typing import Iterable

from modelci.types.do import ProfileResultDO
from .dynamic_profile_result_bo import DynamicProfileResultBO
from .static_profile_result_bo import StaticProfileResultBO


class ProfileResultBO(object):
    """Profiling result business object.
    """

    def __init__(
            self,
            static_result: StaticProfileResultBO,
            dynamic_results: Iterable[DynamicProfileResultBO] = None
    ):
        """
        Initializer.

        Args:
            static_result (StaticProfileResultBO): static profiling result business object.
            dynamic_results (Optional[Iterable[DynamicProfileResultBO]]): a list of dynamic profiling result business
                object. Default to an empty list.
        """
        self.static_result = static_result
        if dynamic_results is None:
            self.dynamic_results = []
        else:
            self.dynamic_results = dynamic_results

    def to_profile_result_po(self):
        """Convert to profile result plain object for persistence.
        """
        # convert static profiling result
        if self.static_result is not None:
            spr_po = self.static_result.to_static_profile_result_po()
        else:
            spr_po = None

        # convert dynamic profiling result
        if self.dynamic_results is not None:
            dpr_pos = list(map(DynamicProfileResultBO.to_dynamic_profile_result_po, self.dynamic_results))
        else:
            dpr_pos = list()

        # create profiling result business object
        pr_po = ProfileResultDO(static_profile_result=spr_po, dynamic_profile_results=dpr_pos)

        return pr_po

    @staticmethod
    def from_profile_result_po(pr_po: ProfileResultDO):
        """Create business object form plain object.
        """
        spr = StaticProfileResultBO.from_static_profile_result_po(pr_po.static_profile_result)
        # convert to DynamicProfileResultBO
        if pr_po.dynamic_profile_results is not None:
            dpr = list(map(DynamicProfileResultBO.from_dynamic_profile_result_po,
                           pr_po.dynamic_profile_results))
        else:
            dpr = list()
        pr = ProfileResultBO(static_result=spr, dynamic_results=dpr)
        return pr
