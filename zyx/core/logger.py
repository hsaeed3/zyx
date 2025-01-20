"""
### zyx.core.logger
"""

import logging

__all__ = ["logger"]


# Logger
logging.basicConfig(level=logging.WARNING, format="%(name)s.%(module)s.%(funcName)s: %(message)s")
logger: logging.Logger = logging.getLogger("zyx")


def debug(value: bool = False) -> None:
    """Sets the library to debug mode

    Example:
        >>> debug()
        >>> from logging import getLogger
        >>> getLogger('zyx').setLevel(logging.DEBUG)
    """
    if value:
        logger.setLevel(logging.DEBUG)
        logger.debug("using logging level: DEBUG")
    else:
        logger.setLevel(logging.WARNING)
