from __future__ import annotations

__all__ = ["console", "logger"]

# hammad saeed - 2024
# zyx._rich
# library level helpers & util
# from rich


# rich print swap
# because it looks pretty
# my library not yours :)
# heh
import builtins as __builtin__
from rich import print as __rich_print
__builtin__.print = __rich_print


# default console & logging setup
# xnano 'styling'
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.theme import Theme
# logging
from rich.logging import RichHandler
import logging
import threading

from typing import Optional


# helper
class XnanoConsole(Console):
    def print_titled_box(self, title: str, content: str):
        panel = Panel(
            Align.center(content),
            title=title,
            expand=False,
            border_style="cyan"
        )
        self.print(panel)


# console helper
def get_xnano_console() -> XnanoConsole:
    return XnanoConsole(
        theme=Theme({
            "info": "dim cyan",
            "warning": "dim yellow",
            "error": "bold red"
        }),
        log_time_format="[%X]",
    
        # library name added in print statements
        record=True
    )


console_lock = threading.Lock()
console_instance = None

def get_console_instance() -> XnanoConsole:
    global console_instance
    with console_lock:
        if console_instance is None:
            console_instance = get_xnano_console()
    return console_instance

console = get_console_instance()


# logger helper
def get_xnano_logger(console: Optional[XnanoConsole] = None) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Use provided console or create a new one if not provided
    if console is None:
        console = get_xnano_console()
    
    # Ensure RichHandler is used with the provided or created console
    if not logger.handlers or not isinstance(logger.handlers[0], RichHandler):
        handler = RichHandler(console=console, rich_tracebacks=True)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    
    return logger


logger_lock = threading.Lock()
logger_instance = None

def get_logger_instance() -> logging.Logger:
    global logger_instance
    with logger_lock:
        if logger_instance is None:
            logger_instance = get_xnano_logger()
    return logger_instance

logger = get_logger_instance()



if __name__ == "__main__":

    logger.info("hello")

    console.print_titled_box("hello", "world")

    logger.fatal("fatal")



