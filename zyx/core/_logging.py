"""zyx.core._logging"""

import logging

from rich import get_console
from rich.logging import RichHandler
from rich import traceback

traceback.install(console=get_console(), width=120, extra_lines=3)

__all__ = ["_get_logger"]


_ZYX_LOGGER: logging.Logger | None = None
"""Singleton logger instance for the `zyx` library."""


class _ZyxRichHandler(RichHandler):
    """Simple logging handler that utilizes `rich` to style the
    logs produced by the `zyx` library.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            console=get_console(),
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_path=False,
            **kwargs,
        )

    def get_level_text(self, record: logging.LogRecord) -> str:
        level_name = record.levelname
        level_no = record.levelno

        if level_no == logging.DEBUG:
            return f"[bright_blue]{level_name}[/bright_blue]"
        elif level_no == logging.INFO:
            return f"[green]{level_name}[/green]"
        elif level_no == logging.WARNING:
            return f"[bold sandy_brown]{level_name}[/bold sandy_brown]"
        elif level_no == logging.ERROR:
            return f"[bold red]{level_name}[/bold red]"
        elif level_no == logging.CRITICAL:
            return f"[bold white on red]{level_name}[/bold white on red]"
        else:
            return f"[bold]{level_name}[/bold]"

    def render_message(self, record: logging.LogRecord, message: str) -> str:
        level_no = record.levelno

        if level_no == logging.CRITICAL or level_no == logging.ERROR:
            return f"[red]{message}[/red]"
        if level_no == logging.DEBUG:
            return f"[dim italic]{message}[/dim italic]"
        elif level_no == logging.WARNING:
            return f"[italic sandy_brown]{message}[/italic sandy_brown]"
        elif level_no == logging.INFO:
            return f"[italic green]{message}[/italic green]"
        else:
            return message


def _get_logger(
    name: str | None = None,
    level: int | str | None = None,
) -> logging.Logger:
    """Internal helper function that returns a styled logger instance."""
    if level is None:
        level = logging.NOTSET
    if isinstance(level, str):
        level = logging.getLevelNamesMapping()[level.upper()]

    global _ZYX_LOGGER
    logger_name = name or "zyx"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Only add handler to root 'zyx' logger
    if logger_name == "zyx":
        if not any(isinstance(h, RichHandler) for h in logger.handlers):
            rich_handler = _ZyxRichHandler()
            rich_handler.setLevel(logging.NOTSET)
            logger.addHandler(rich_handler)
            logger.propagate = False

        if _ZYX_LOGGER is None:
            _ZYX_LOGGER = logger
    else:
        # All child loggers propagate to root
        logger.propagate = True
        # Do not add handler to child loggers

    if _ZYX_LOGGER is None:
        _ZYX_LOGGER = _get_logger("zyx", level)

    return logger
