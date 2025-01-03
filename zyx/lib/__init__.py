# [zyx.lib]
# ===================================================================

__all__ = ["utils"]


# utils is the only module exported at the init level of `zyx.lib`
from ._utils import Utils

utils = Utils()  # type: ignore
