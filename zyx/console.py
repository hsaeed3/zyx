# zyx.console
# singleton console object with helper methods
# used internally

# builtins -- zyx swaps out print with rich.print
import builtins
from rich import print
# console & ext imports
from rich.console import Console as RichConsole
from rich.progress import Progress, TextColumn, BarColumn
# imports
from contextlib import contextmanager
import inspect


# print swap
builtins.print = print


# console resource
class Console(RichConsole):

    """Singleton Console Resource for zyx"""

    # super
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # message printer for all 'verbose' message for methods
    def message(self, message : str):
        frame = inspect.currentframe().f_back
        module = frame.f_globals["__name__"]
        self.print(f"⚛️ [bold plum3]ZYX[/bold plum3] | [cyan]{module}[/cyan] | [white italic]{message}[/white italic]")


    # warning method
    # no exception -- exceptions raised by zyx.exception
    def warning(self, message : str):
        frame = inspect.currentframe().f_back
        module = frame.f_globals["__name__"]
        self.print(f"⚠️ [bold yellow]ZYX WARNING[/bold yellow] | [yellow]{module}[/yellow] | [yellow italic]{message}[/yellow italic]")


    # context manager for constant progress loader
    @contextmanager
    def progress(self, message : str, *args, **kwargs) -> Progress:
        with Progress(
            TextColumn("[progress.description]{task}"),
            BarColumn(bar_width=None),
            *args,
            **kwargs
        ) as progress:
            task = progress.add_task(description=message, total=None)

            # try block to remove loader on exit
            try:
                yield progress, task
            finally:
                progress.remove_task(task)
