"""
🔎 ### zyx.lib._logger

Contains the basic logger config used in `zyx`. Uses utilizes
Rich.
"""

import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install


console : Console = Console()
"""Rich console instance."""
install(console=console)


# ------------------------------------------------------------------------
# [Global Variables & Flags]
# ------------------------------------------------------------------------


_zyx_init = False
"""Flag for checking if logging has been initialized."""


_zyx_logger : logging.Logger = None
"""The logger instance for `zyx`."""


# ------------------------------------------------------------------------
# [Setup]
# ------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    """
    Initializes logging for `zyx`.
    """
    global _zyx_init, _zyx_logger
    
    if _zyx_init and _zyx_logger is not None:
        return _zyx_logger
    
    # Create config
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s.%(module)s.%(funcName)s: %(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False, show_level=False, markup=True)],
    )
    # get logger
    _zyx_logger = logging.getLogger("zyx")
    # set flag
    _zyx_init = True
    # return logger
    return _zyx_logger


setup_logging()


# ------------------------------------------------------------------------
# [Debug]
# ------------------------------------------------------------------------


def debug(value : bool) -> None:
    """
    Sets the library and core logging level to DEBUG.
    """
    global _zyx_logger
    if not _zyx_logger:
        setup_logging()
    
    if value:
        _zyx_logger.setLevel(logging.DEBUG)
        _zyx_logger.debug("using logging level: DEBUG")
    else:
        _zyx_logger.setLevel(logging.WARNING)
