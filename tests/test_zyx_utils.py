"""
Core utility tests for zyx.
"""

import pytest
from zyx import _utils
from pathlib import Path
import logging
import os


# ===================================================================
# [Initialization Check]
# ===================================================================

def test_zyx_initialized() -> None:
    
    """
    Tests library & cache initialization.
    """
    
    # Initialize
    _utils.initialize_zyx()
    
    # Check
    assert _utils.ZYX_CACHE_DIR.exists()
    assert _utils.ZYX_CONFIG_FILE.exists()
    assert _utils.zyx_logger is not None
    assert isinstance(_utils.zyx_logger, logging.Logger)

