"""
### zyx.logging

Primary module for library verbosity & logging configuration as well
as the resource for printing/getting inputs w/ rich text in the console.
"""

from __future__ import annotations

# [Imports]
from enum import Enum
import inspect
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install


__all__ = [
    # [Flags]
    "get_verbosity_level",
    "verbose",
    "debug",
    # [Helpers]
    "verbose_print",
    "warn",
    # [Logger]
    "logger",
    # [Console]
    "console",
    # [Exceptions]
    "ZyxException",
]


# flags
_is_zyx_logging_initialized = False
"""Internal flag to check if logging has been initialized."""


# traceback
console = Console()
install(console=console)


# exception
class ZyxException(Exception):
    """Base exception raised by the `zyx` package."""


# ========================================================================
# [Global Flag Config]
# ========================================================================


# [Class]
class _LoggingConfig:
    """Configuration manager for verbosity in the `zyx` package."""

    _instance = None

    # [Attributes]
    verbosity_level: int = 0
    """
    The current verbosity level of the `zyx` package. Most main modules in the library
    will provide both `verbose` and `debug` arguments that will be used to set this flag.

    - `Level 0`: Silent (level WARNING)
    - `Level 1`: Verbose (level WARNING + simple verbose CLI outputs & logging through the `rich` library)
    - `Level 2`: DEBUG (level DEBUG + debug logs & verbose CLI outputs w/ rich)
    """

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_LoggingConfig, cls).__new__(cls)

            # [Load Default Config Level at 0 (Silent)]
            cls._instance.verbosity_level = 0
        return cls._instance

    # [Methods]
    def set_verbosity_level(self, level: int) -> None:
        """Set the verbosity level of the `zyx` package."""

        if level not in [0, 1, 2]:
            raise ZyxException(f"Invalid verbosity level: {level}. Must be 0, 1, or 2.")

        # Update the logger's level directly
        if level == 0:
            logger.setLevel(logging.WARNING)
        elif level == 1:
            logger.setLevel(logging.WARNING)
        elif level == 2:
            logger.setLevel(logging.DEBUG)

        # [Set Verbosity Level]
        self.verbosity_level = level

    def get_verbosity_level(self) -> int:
        """Get the current verbosity level of the `zyx` package."""
        return self.verbosity_level


# [Singleton]
_logging_config = _LoggingConfig()


# ========================================================================
# [Logging & Console Outputs]
# ========================================================================


# [Logging Setup]
def _setup_logging() -> logging.Logger:
    """
    Setup logging configuration for the `zyx` package.

    This function initializes the logging configuration with a specific format and handlers.
    It ensures that the logging is only initialized once by using a global flag.

    Returns:
        logging.Logger: The configured logger for the `zyx` package.
    """
    global _is_zyx_logging_initialized
    if _is_zyx_logging_initialized:
        return logging.getLogger("zyx")

    # [Setup Logging]
    # [Remove Existing Handlers]
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # [Config]
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s.%(module)s.%(funcName)s - DEBUG - %(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False, show_level=False, markup=True)],
    )

    # [Create Logger]
    logger = logging.getLogger("zyx")

    # [Check External Logging Level]
    if logging.root.level == logging.DEBUG:
        _logging_config.set_verbosity_level(2)

    # [Set Flag]
    _is_zyx_logging_initialized = True

    return logger


# [Logger Singleton for Internal Use]
logger = _setup_logging()


# ========================================================================
# [Flag Helpers]
# ========================================================================


# [Get Verbosity Level]
def get_verbosity_level() -> int:
    """Get the current verbosity level of the `zyx` package."""
    return _logging_config.get_verbosity_level()


# [Set Verbose Mode]
def verbose(verbose: bool) -> None:
    """
    Helper method to enable `verbose` mode for zyx, providing simple but useful &
    pretty CLI outputs through the `rich` library.
    """
    if verbose:
        _logging_config.set_verbosity_level(1)
        verbose_print(
            "[italic]Verbose logging[/italic] enabled! [bold]:)[/bold] [tan]Behold[/tan] [sky_blue3]pretty[/sky_blue3] [orchid2]colors[/orchid2]!"
        )
    else:
        _logging_config.set_verbosity_level(0)


# [Verbose Print]
def verbose_print(*args, **kwargs) -> None:
    """
    Prints the provided arguments to the console using the `rich` library.
    Only prints if zyx_verbose is True.
    """
    if get_verbosity_level() >= 1:
        # Get the caller's frame
        caller_frame = inspect.currentframe().f_back
        # Get the module name from the file path
        module_name = caller_frame.f_globals["__name__"]
        if module_name == "__main__":
            # If running as main, get the module name from the file path
            module_name = "zyx._logging"

        console.print(
            "[bold light_sky_blue3]zyx[/bold light_sky_blue3] | "
            f"[dim italic]{module_name}[/dim italic] | " + " ".join(str(arg) for arg in args),
            **kwargs,
        )


# [Set Debug Mode]
def debug(debug: bool) -> None:
    """
    Helper method to enable `debug` mode for zyx, providing DEBUG level logging.
    """
    if debug:
        _logging_config.set_verbosity_level(2)
        logger.debug("using DEBUG level logging")
    else:
        _logging_config.set_verbosity_level(0)


# [Warning Helper]
def warn(message: str) -> None:
    """
    Prints a warning message to the console using the `rich` library.
    """
    console.print(f"[bold orange3]WARNING[/bold orange3]: [italic yellow]{message}[/italic yellow]")
