"""
utils tests for zyx.lib._utils
"""

import pytest
from zyx.lib import _utils as utils


# ===================================================================
# Console Test
# ===================================================================

def test_lib_utils_rich_console():
    
    from rich.console import Console
    
    utils.console.print("This is a [bold red]test![/bold red]")
    
    assert isinstance(utils.console, Console)
    
    
def test_lib_utils_logger_and_flags():
    
    utils.logger.debug("This is a debug message")
    
    utils.set_zyx_debug(True)
    
    utils.logger.debug("This is a debug message")
    
    assert utils.zyx_debug is True
    assert utils.zyx_verbose is False
    assert utils.logger is not None
    
    
if __name__ == "__main__":
    test_lib_utils_rich_console()
    test_lib_utils_logger_and_flags()
    
    