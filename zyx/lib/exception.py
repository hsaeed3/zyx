# zyx.lib.exception
# base exception class used internally

import inspect
from rich.console import Console
from rich.traceback import install


# exception console
console = Console()


# install rich traceback
install(console=console)


# zyxexception
class ZYXException(Exception):

    """Base exception class for zyx."""

    def __init__(
            self,
            message : str
    ):
        
        # self.message is implemented with rich color tags
        # zyx uses rich .print as a builtin print method
        # console handles all other printing
        frame = inspect.currentframe().f_back
        module = frame.f_globals["__name__"]
        self.message = f"[red]ZYX [italic bold white]{module}[/italic bold white] Exception:[/red] {message}"


    def __str__(self):
        # minimal message sent as non rich output
        return "ZYX Exception Occured"
    

    # prints exception message on exit (to use rich formatting)
    def __del__(self):
        console.print(self.message)


# test
if __name__ == "__main__":
    raise ZYXException("This is a test exception")