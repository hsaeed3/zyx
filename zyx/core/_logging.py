"""zyx.core._logging"""

import logging
import sys
from typing import Optional

from rich import traceback, get_console

traceback.install(console=get_console(), width=120, extra_lines=3)

__all__ = ["_get_logger"]


_ZYX_LOGGER: logging.Logger | None = None
"""Singleton logger instance for the `zyx` library."""


_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_BOLD_OFF = "\x1b[22m"
_ITALIC = "\x1b[3m"
_ITALIC_OFF = "\x1b[23m"
_DIM = "\x1b[2m"
_DIM_OFF = "\x1b[22m"
_FG_COLORS = {
    logging.DEBUG: "\x1b[38;5;75m",  # bright blue
    logging.INFO: "\x1b[38;5;84m",  # bright green
    logging.WARNING: "\x1b[38;2;255;175;95m",  # bright orange
    logging.ERROR: "\x1b[38;5;203m",  # bright red/orange
    logging.CRITICAL: "\x1b[38;5;207m",  # bright magenta
}
_DIM_COLORS = {lvl: f"{_DIM}{color}" for lvl, color in _FG_COLORS.items()}


def _format_level_tag(levelno: int) -> str:
    """Format the level name with appropriate styling."""
    level_name = logging.getLevelName(levelno)
    color = _FG_COLORS.get(levelno, "")

    if levelno == logging.WARNING:
        return f"{_ITALIC}{_BOLD}{color}{level_name}{_RESET}"
    elif levelno == logging.CRITICAL:
        return f"{_BOLD}{color}{level_name}{_RESET}"
    else:
        return f"{_BOLD}{color}{level_name}{_RESET}"


class _StyledFormatter(logging.Formatter):
    """Custom formatter with ANSI styling for console output."""

    def __init__(self, fmt=None, datefmt=None, style="%", validate=True):
        super().__init__(fmt, datefmt, style, validate)

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with styled output."""
        level_tag = _format_level_tag(record.levelno)
        logger_name = f"{_BOLD}{record.name}{_BOLD_OFF}"
        message = record.getMessage()

        # Apply dim styling to message based on level
        dim_color = _DIM_COLORS.get(record.levelno, _DIM)
        styled_message = f"{dim_color}{message}{_RESET}{_DIM_OFF}"

        # Include exception info if present
        formatted = f"{level_tag} {logger_name}: {styled_message}"

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            if formatted[-1:] != "\n":
                formatted += "\n"
            formatted += record.exc_text

        if record.stack_info:
            if formatted[-1:] != "\n":
                formatted += "\n"
            formatted += self.formatStack(record.stack_info)

        return formatted


class _ConsoleHandler(logging.StreamHandler):
    """Custom handler that outputs to stdout with proper encoding."""

    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record with error handling."""
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def _get_logger(
    name: Optional[str] = None,
    level: Optional[int | str] = None,
) -> logging.Logger:
    """
    Get or create a styled logger instance.

    Args:
        name: Logger name. Defaults to 'zyx'.
        level: Logging level (int or string like 'INFO', 'DEBUG').
               Defaults to NOTSET.

    Returns:
        Configured logger instance.
    """
    global _ZYX_LOGGER

    # Normalize level
    if level is None:
        level = logging.NOTSET
    elif isinstance(level, str):
        level_map = {
            "NOTSET": logging.NOTSET,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.CRITICAL,
        }
        level = level_map.get(level.upper(), logging.NOTSET)

    logger_name = name or "zyx"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Only configure the root 'zyx' logger with a handler
    if logger_name == "zyx":
        # Check if handler already exists to avoid duplicates
        if not any(isinstance(h, _ConsoleHandler) for h in logger.handlers):
            handler = _ConsoleHandler()
            handler.setLevel(logging.NOTSET)
            handler.setFormatter(_StyledFormatter())
            logger.addHandler(handler)

        # Prevent propagation to root logger
        logger.propagate = False

        # Set global reference
        if _ZYX_LOGGER is None:
            _ZYX_LOGGER = logger
    else:
        # Child loggers propagate to parent
        logger.propagate = True

    # Ensure root logger exists
    if _ZYX_LOGGER is None:
        _ZYX_LOGGER = _get_logger("zyx", level)

    return logger
