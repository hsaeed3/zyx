# zyx.shared.types.exception
# exception (UH OHHHHH)

from __future__ import annotations


__all__ = [
    "ZyxError",
    "Yikes"
]


from . import _rich as utils
import traceback


class ZyxError(Exception):
    """
    Generic exception type for Zyx.
    """

    def __init__(self, message: str = "An error occurred", name : str = "Zyx"):
        self.name = name
        self.message = message
        self.traceback = traceback.format_exc()
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"

    def __del__(self):
        utils.console.print(f"📝 [bold red]Traceback:[/bold red] {self.traceback}")
        utils.console.print(f"🙈 [red]OhNoo [bold dark_red]{self.name}[/bold dark_red]Error!!![/red] {self.message}")


class Yikes:
    """
    Warning class for Zyx
    """

    def __init__(self, message: str):
        self.message = message
        utils.console.print(f"⚠️ [bold yellow]Yikes![/bold yellow] [yellow]Warning:[/yellow] {message}")

    def __str__(self) -> str:
        return self.message
    

# function exceptions
ClassifierError = ZyxError
CoderError = ZyxError
ExtractorError = ZyxError
FunctionError = ZyxError
GeneratorError = ZyxError
PatcherError = ZyxError
PrompterError = ZyxError
PlannerError = ZyxError
QaError = ZyxError
QueryError = ZyxError
SelectorError = ZyxError
SolverError = ZyxError
ValidatorError = ZyxError
# agents
AgentsError = ZyxError
    

if __name__ == "__main__":

    # warning
    Yikes("This is a test")

    # error
    raise ZyxError(name="Zyx", message="This is a test")
