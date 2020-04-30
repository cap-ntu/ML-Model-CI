from typing import Optional

from ..po.static_profile_result_po import StaticProfileResultPO


class StaticProfileResultBO(object):
    def __init__(self, parameters: int, flops: int, memory: int, mread: int, mwrite: int, mrw: int):
        """Initializer.

        Args:
            parameters (int): number of parameters.
            flops (int): total floating point operations to run the model.
            memory (int): memory occupation in Byte.
            mread (int): memory read size.
            mwrite (int): memory write size.
            mrw (int): memory read write size.
        """
        self.parameters = parameters
        self.flops = flops
        self.memory = memory
        self.mread = mread
        self.mwrite = mwrite
        self.mrw = mrw

    def to_static_profile_result_po(self):
        """Convert business object to plain object.
        """
        static_profile_result_po = StaticProfileResultPO(
            parameters=self.parameters, flops=self.flops,
            memory=self.memory, mread=self.mread, mwrite=self.mwrite,
            mrw=self.mrw)
        return static_profile_result_po

    @staticmethod
    def from_static_profile_result_po(spr_po: Optional[StaticProfileResultPO]):
        """Create business object from a plain object.

        Args:
            spr_po (Optional[StaticProfileResultPO]): static profiling result plain object. Default to None.
        """
        # spr_po nullable
        if spr_po is None:
            return None

        spr = StaticProfileResultBO(parameters=spr_po.parameters, flops=spr_po.flops, memory=spr_po.memory,
                                    mread=spr_po.mread, mwrite=spr_po.mwrite, mrw=spr_po.mrw)
        return spr
