"""zyx.core.exceptions"""

from rich import get_console
from rich import traceback

traceback.install()

__all__ = [
    "rich_warning",
    "ZyxError",
    "ModelProviderException",
    "ModelProviderInferenceException",
    "ModelAdapterException",
]


def rich_warning(message: str, name: str | None = None) -> None:
    """Used to display simple warning messages to the user, does not perform any
    logging operations."""
    if not name:
        get_console().print(
            f"[bold sandy_brown]{r'\[WARNING]'}[/bold sandy_brown] | [italic yellow]{message}[/italic yellow]"
        )
    else:
        content = f"[bold sandy_brown]{r'\[WARNING - [italic]{name}[/italic]]'}[/bold sandy_brown] | [italic yellow]{message}[/italic yellow]"
        get_console().print(content.format(name=name, message=message))


class zyxError(Exception):
    """Base exception class for all library specific exceptions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class InstructorLibraryException(zyxError):
    """Exception type used to wrap any errors that originate directly
    from the `instructor` package.

    Instructor is the most fundamental dependency within `zyx`, and all
    language model / run logic is dependent on various components within
    it.
    """

    def __init__(self, exception: Exception, message: str = "", *args, **kwargs):
        self.exception = exception
        self.message = message
        super().__init__(message, *args, **kwargs)

    def __str__(self) -> str:
        content = (
            "Error raised from the [italic]`instructor`[/italic] library: \n"
            + str(self.exception)
            + "\n"
            if str(self.exception)
            else ""
        )
        if self.message:
            content += f"\n{self.message}"
        return content


class ModelProviderException(zyxError):
    """Base exception class for all associated AI model provider errors."""

    pass


class ModelProviderInferenceException(zyxError):
    """Exception raised if 'automatic' model provider inference fails
    or any other provider-specific errors occur."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        message: str = "",
        *args,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.message = message
        super().__init__(message, *args, **kwargs)

    def __str__(self) -> str:
        content = (
            "Failed to automatically infer or configure a model provider for the given model and (if applicable) Base URL:\n"
            f"  - Model: {self.model}\n"
            f"  - Base URL: {self.base_url if self.base_url else 'None'}\n"
        )
        if self.message:
            content += f"\nAdditional error message: {self.message}"
        return content

    def __repr__(self) -> str:
        return f"ModelProviderInferenceException(model={self.model}, base_url={self.base_url}, message={self.message})"


class ModelAdapterException(zyxError):
    """Base exception for all model adapter errors."""

    def __init__(self, name: str = "adapter", message: str = "", *args, **kwargs):
        self.name = name
        self.message = message
        super().__init__(message, *args, **kwargs)

    def __str__(self) -> str:
        return f"[{self.name.upper()} ADAPTER]: {self.message}"

    def __repr__(self) -> str:
        return f"ModelAdapterException(name={self.name}, message={self.message})"


class LiteLLMModelAdapterError(ModelAdapterException):
    """Exception for LiteLLMModelAdapter errors."""

    def __init__(self, message: str, *args, **kwargs):
        super().__init__(name="litellm", message=message, *args, **kwargs)


class OpenAIModelAdapterError(ModelAdapterException):
    """Exception for OpenAIModelAdapter errors."""

    def __init__(self, message: str, *args, **kwargs):
        super().__init__(name="openai", message=message, *args, **kwargs)
