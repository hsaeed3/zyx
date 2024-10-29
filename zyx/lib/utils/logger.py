"""Zyx logger.

This module provides a custom logger for the Zyx project.
It uses rich for colorized output and swaps out the built-in
`print` function with `rich_print`.

Author: Hammad Saeed
"""

import builtins
import logging


from rich.logging import RichHandler
from rich import print as rich_print


builtins.print = rich_print


def get_logger(module_name: str = __name__, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(module_name)

    # Remove any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logging level
    logger.setLevel(level)

    # Create a RichHandler
    console_handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
    )

    # Set the format for the handler
    FORMAT = "[bold italic red]%(name)s[/bold italic red] - [italic dim]%(funcName)s[/italic dim] - %(message)s"
    console_handler.setFormatter(logging.Formatter(FORMAT))

    # Add the handler to the logger
    logger.addHandler(console_handler)

    # Ensure all messages are propagated
    logger.propagate = False

    return logger


logger = get_logger("zyx")
