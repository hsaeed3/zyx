"""
zyx.lib

Contains logging/config utils, flags & other core resources used throughout `zyx`.
"""

__all__ = [
    # [Core Utils & Flags]
    "utils",
    "zyx_verbose",
    "zyx_debug",
]


# ===================================================================
# [Imports]
# ===================================================================

# [Core Utils & Flags]
from . import _utils as utils

# Retrieve Flags, so that if they are set here, they are set globally as well
global zyx_verbose
global zyx_debug
zyx_verbose: bool = utils.zyx_verbose
"""Modules will provide printed console outputs & simple information."""
zyx_debug: bool = utils.zyx_debug
"""Modules will provide extensive & detailed debug information."""
