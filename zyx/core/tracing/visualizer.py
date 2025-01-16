"""
### zyx.core.tracing.visualizers

Contains modules using the Rich library for pretty pretty words and shapes
and colors :) Also sets up logging w/ rich.

Progress bars!
And spinners, trees, columns so many things!

Also used for the .visualize() method for graphs in the console.
"""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.traceback import install
from rich.tree import Tree

import logging
from threading import Lock
from typing import Dict, Union, Optional
from ..types.tracing import TracingStyles


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

# singleton console
console = Console()

# logger
_zyx_logger: logging.Logger = None

# traceback
install()


# helper function for logging
def setup_logging() -> logging.Logger:
    """
    Initializes logging for `zyx`.
    """
    global _zyx_logger
    if _zyx_logger is not None:
        return _zyx_logger
    # create new root logger
    logger = logging.getLogger("zyx")
    # configure logging with rich handler
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s.%(module)s.%(funcName)s: %(message)s",
        handlers=[
            RichHandler(
                console=console,
                show_time=False,
                show_level=False,
                markup=True,
            )
        ],
    )
    # set flag
    _zyx_logger = logger
    # return logger
    return logger


def debug(value: bool) -> None:
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


# initialize logging & get logger as 'logger'
logger = setup_logging()


# -----------------------------------------------------------------------------
# Tracing Visualizer
# -----------------------------------------------------------------------------


class TracingVisualizer:
    """
    A global library singleton, used as the core resource
    for rich console inputs, through zyx.core.tracing
    """

    _instance: TracingVisualizer = None
    """Singleton"""

    _lock: Lock = Lock()
    """Lock for thread safety"""

    _styles: TracingStyles = TracingStyles()
    """Singleton for the TracingStyles class"""

    # initializer
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._live = None  # Rich.Live instance
                cls._logs: Dict[str, Tree] = {}  # Logs or loaders per entity (keyed by name)
                cls._tasks: Dict[str, Union[Spinner, Progress]] = {}  # Active loaders
                cls._is_active = False  # Tracks if live tracing is enabled
            return cls._instance

    # -------------------------------------------------------------------------
    # [Live Rendering Helpers]
    # -------------------------------------------------------------------------

    def print_header(self):
        """Print the tracing enabled header."""
        color = self._styles.randomcolor()
        console.print(
            f"🔍 {self._styles.title('zyx')} [bold {color}]Tracing enabled![/bold {color}] - [italic {color}]events will be logged to console[/italic {color}]"
        )

    def _render_tree(self):
        """Renders a Rich.Tree dynamically based on the current logs & state."""
        # Only render tree if there are logs
        if not self._logs:
            return ""

        root = Tree(f"{self._styles.title('🔎 Trace')}")
        for entity, tree in self._logs.items():
            root.add(tree)
        return root

    def _render_table(self):
        """Renders a Rich.Table dynamically for multiple running traced modules."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Entity", style="cyan", no_wrap=True)
        table.add_column("Logs", style="white")

        for entity, log in self._logs.items():
            table.add_row(entity, log)

        return table

    def _refresh_live(self):
        """Refreshes the live rendering of the tracer."""
        if self._is_active and self._live is not None:
            if len(self._logs) > 1:
                self._live.update(self._render_table())
            else:
                self._live.update(self._render_tree())

    # -------------------------------------------------------------------------
    # [Toggles / Controls]
    # -------------------------------------------------------------------------

    def enable_live(self):
        """Starts the rich.Live instance, if not already active."""
        if not self._is_active:
            with self._lock:
                if not self._is_active:
                    # Don't render empty tree until first event
                    self._live = Live(
                        "",  # Start with empty content
                        console=console,
                        refresh_per_second=4,
                    )
                    self._live.start()
                    self._is_active = True
                    logger.debug("enabled tracing visualizer")

    def disable_live(self):
        """Stops the rich.Live instance, if active."""
        if self._is_active:
            with self._lock:
                self._live.stop()
                self._tasks.clear()
                self._live = None
                self._is_active = False
                logger.debug("disabled tracing visualizer")

    def log(self, entity: str, message: str):
        """Logs a message to the tracer."""
        with self._lock:
            if entity not in self._logs:
                self._logs[entity] = Tree(f"{self._styles.subtitle(entity)}")
            # Format message consistently
            formatted_msg = f"{message}"
            self._logs[entity].add(formatted_msg)
            self._refresh_live()

    # -------------------------------------------------------------------------
    # [Tasks (Progress Bars & Spinners)]
    # -------------------------------------------------------------------------

    def add_task(
        self,
        entity: str,
        description: str,
        total: Optional[int] = None,
    ):
        """Creates a new 'task' to the tracer, can be displayed with a spinner or progress bar.

        Args:
            entity (str): The name of the entity to add the task to.
            description (str): The description of the task.
            total (Optional[int]): The total number of steps in the task, if applicable.
        """
        with self._lock:
            if entity not in self._logs:
                self._logs[entity] = Tree(f"{self._styles.subtitle(entity)}")
            if total is None:
                spinner = Spinner("dots", text=description)
                self._logs[entity].add(spinner)
                self._tasks[description] = spinner
            else:
                progress = Progress(
                    TextColumn(f"[italic]{description}[/italic]"),
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeRemainingColumn(),
                )
                task_id = progress.add_task(description, total=total)
                self._logs[entity].add(progress)
                self._tasks[description] = (progress, task_id)
            self._refresh_live()

    def update_task(
        self,
        description: str,
        advance: int = 1,
    ):
        """Updates a task's progress."""
        with self._lock:
            if description in self._tasks:
                task = self._tasks[description]
                if isinstance(task, tuple):  # It's a progress bar
                    progress, task_id = task
                    progress.update(task_id, advance=advance)
                self._refresh_live()

    def complete_task(self, description: str):
        """Mark a task as complete and remove it from the tree."""
        with self._lock:
            if description in self._tasks:
                del self._tasks[description]
            self._refresh_live()


# -----------------------------------------------------------------------------
# [Singleton]
# -----------------------------------------------------------------------------

tracing_visualizer = TracingVisualizer()
"""Singleton for the TracingVisualizer class"""
