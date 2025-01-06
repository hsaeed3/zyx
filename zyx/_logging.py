"""
zyx._logging

This module contains the various utility resources, variables & configurations
used internally by the `zyx` library.
"""

__all__ = [
    # [Flags]
    "zyx_verbose",
    "zyx_debug",
    # [Console]
    "console",
    # [Logger]
    "logger",
    # [Helpers]
    "set_zyx_verbose",
    "set_zyx_debug",
    "get_logger",
    # [Exceptions]
    "ZyxException",
    "warn",
]


# [Imports]
from typing import List, Dict, Any, Union
import builtins
import logging
import json
import os
from pathlib import Path
from pydantic import BaseModel
from functools import wraps

from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.traceback import install


# ===================================================================
# [Singletons]
# ===================================================================

# [Rich Console]
console: Console = Console(markup=True)
"""
Rich console instance.
"""

# [Rich Traceback]
install(console=console)

# NOTE:
# zyx uses rich.print() at builtins level...
# ill remove this probably, it just makes it prettier
builtins.print = rprint

# [Logger]
logger: logging.Logger = logging.getLogger("zyx")
# Initialize with DEBUG level, but handler will control actual output
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()

handler = RichHandler(
    console=console,
    markup=True,
    show_time=True,
    show_level=True,
)
# Default to WARNING until verbose/debug is set
handler.setLevel(logging.WARNING)
logger.addHandler(handler)


# ===================================================================
# [Library Level Flags]
# ===================================================================

# [Verbosity]
zyx_verbose: bool = False
"""Modules will provide printed console outputs & simple information."""

# [Debug]
zyx_debug: bool = False
"""Modules will provide detailed debug information."""


# [Helpers]
def set_zyx_verbose(verbose: bool) -> None:
    """
    Sets the verbose flag and adjusts logging level.

    Args:
        verbose: If True, sets logging to INFO level
    """
    global zyx_verbose
    zyx_verbose = verbose
    if verbose:
        handler.setLevel(logging.INFO)
    else:
        handler.setLevel(logging.WARNING)


def set_zyx_debug(debug: bool) -> None:
    """
    Sets the debug flag and adjusts logging level.

    Args:
        debug: If True, sets logging to DEBUG level
    """
    global zyx_debug
    zyx_debug = debug
    if debug:
        handler.setLevel(logging.DEBUG)
        console.print(f"[green]DEBUG[/green] mode enabled for {Styles.zyx()} logging")
    else:
        handler.setLevel(logging.WARNING if not zyx_verbose else logging.INFO)


# ===================================================================
# [Script Specific Flags]
# ===================================================================

# [Library Initialization Check]
zyx_initialized: bool = False
"""Internal flag that validates if the library has been initialized.
Used to prevent multiple initializations."""


# ===================================================================
# [Logging]
# ===================================================================


def get_logger() -> logging.Logger:
    """
    Retrieves the zyx logger instance.
    """
    return logger


# [Logging Style Helpers]
class Styles:
    """
    Styles for zyx logging.

    These are helper functions that just make it easier to add styles
    to specific statements or phrases when creating log statements.

    In other words it looks pretty.
    """

    @staticmethod
    def zyx() -> str:
        """Library Name"""
        return "[bold italic light_sky_blue3]zyx[/bold italic light_sky_blue3]"

    @staticmethod
    def module(name: Any) -> str:
        """Module and/or 'main' name"""
        return f"[bold light_coral]{name if isinstance(name, str) else str(name)}[/bold light_coral]"

    @staticmethod
    def debug(message: Any) -> str:
        return f"[deep_sky_blue2]DEBUG[/deep_sky_blue2]: [dim]{message if isinstance(message, str) else str(message)}[/dim]"

    # [Console Display Helpers] ====================================

    @staticmethod
    def typed_item(item: Union[List[Any], Dict[str, Any], BaseModel]) -> Panel:
        """
        Displays specific types neatly in a panel.
        """
        if isinstance(item, list):
            # Create Table
            table = Table(show_header=True, header_style="bold")
            # Add Items
            for i in item:
                table.add_row(i)
            # Return
            return Panel(table, title="List", border_style="dim")
        elif isinstance(item, dict):
            # Create Table
            table = Table(show_header=True, header_style="bold")
            table.add_column("Key", style="bold")
            table.add_column("Value")

            # Add Items
            for key, value in item.items():
                table.add_row(str(key), str(value))

            # Return
            return Panel(table, title="Dictionary", border_style="dim")

        elif isinstance(item, BaseModel):
            # Convert model to dict and display
            model_dict = item.model_dump()
            return Styles.display_item(model_dict)
        else:
            raise TypeError(f"Cannot display item of type {type(item)}")


# ===================================================================
# [Base Exception]
# ===================================================================


# Handled by rich.traceback automatically, just needs to be called
class ZyxException(Exception):
    """Base exception class used in zyx."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        # Log
        logger.error(message)


# Warnings
def warn(message: str) -> None:
    """Logs a warning message."""
    logger.warning(message)


# ===================================================================
# [Core Util Functions]
# ===================================================================


# [Initializer]
def initialize_zyx():
    """
    Initializes the zyx library by ensuring resources & cache
    directory.
    """
    # Initialization Check
    global zyx_initialized
    if zyx_initialized:
        return

    # Retrieve Global Flags
    global zyx_verbose
    global zyx_debug

    # Set Initialized Flag
    zyx_initialized = True


if __name__ == "__main__":
    initialize_zyx()

    logger.warning("This is a warning message")