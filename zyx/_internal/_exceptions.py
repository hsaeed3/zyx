"""zyx._internal._exceptions"""

from __future__ import annotations

from typing import Any

__all__ = (
    "ZYXError",
    "ZYXWarning",
    # ------- CORE.PROCESSING --------
    "ProcessingError",
    "SchemaValidationError",
    "SchemaExecutionError",
    "TextError",
    "ChunkingError",
    # ------- CORE.MODELS --------
    "ModelClientError",
    "ModelDefinitionError",
    "ModelRequestError",
    # ------- CORE.INTERFACES --------
    "MakerError",
    "ThingError",
    "GuardrailError",
    "StuffError",
)


class ZYXError(Exception):
    """Base error class for all custom exceptions raised by the
    `zyx` library.
    """

    pass


class ZYXWarning(Warning):
    """Base warning class for all custom warnings raised by the
    `zyx` library.
    """

    pass


# ------- CORE.PROCESSING --------


class ProcessingError(ZYXError):
    """Base error class for all custom exceptions raised by the
    `zyx.core.processing` module.
    """

    pass


class SchemaValidationError(ProcessingError):
    """Exception raised when schema validation fails.

    This exception is used throughout the schema processing system
    to indicate that data did not match the expected schema.
    """

    def __init__(
        self, message: str, errors: list[dict[str, Any]] | None = None
    ):
        super().__init__(message)
        self.errors = errors or []


class SchemaExecutionError(ProcessingError):
    """Exception raised when schema execution fails.

    This exception is used when a function schema execution fails,
    such as when required parameters are missing.
    """

    def __init__(
        self, message: str, errors: list[dict[str, Any]] | None = None
    ):
        super().__init__(message)
        self.errors = errors or []


class TextError(ProcessingError):
    """Exception raised when a text operation fails.

    This exception is used for errors during text operations,
    such as invalid parameters or generation failures.
    """

    pass


class ChunkingError(TextError):
    """Exception raised when a chunking operation fails.

    This exception is used for errors during chunking operations,
    such as invalid parameters or generation failures.
    """

    pass


# ------- CORE.MODELS --------


class ModelClientError(ZYXError):
    """Exception raised specifically if a model client fails to generate
    a response."""

    def __init__(
        self,
        message: str,
        client: str,
        model: str | None = None,
    ) -> None:
        self.message = message
        self.client = client
        self.model = model

        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message}: {self.client} (model: {self.model})"


class ModelDefinitionError(ZYXError):
    """Exception raised when the 'definition' or initialization of a model
    and its associated components fails."""

    def __init__(self, message: str, model: str):
        self.message = message
        self.model = model
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message}\nModel: {self.model}"


class ModelRequestError(ZYXError):
    """Exception raised when a model request fails."""

    def __init__(self, message: str, model: str):
        self.message = message
        self.model = model
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message}\nModel: {self.model}"


# ------- CORE.INTERFACES --------


class MakerError(ZYXError):
    """Exception raised when a maker operation fails.

    This exception is used for errors during make/edit operations,
    such as invalid parameters or generation failures.
    """

    pass


class ThingError(ZYXError):
    """Exception raised when a thing operation fails.

    This exception is used for errors during thing operations,
    such as invalid parameters or generation failures.
    """

    pass


class GuardrailError(ZYXError):
    """Exception raised when a guardrail operation fails.

    This exception is used for errors during guardrail operations,
    such as invalid parameters or generation failures.
    """

    pass


class StuffError(ZYXError):
    """Exception raised when a stuff operation fails.

    This exception is used for errors during stuff operations,
    such as invalid parameters or generation failures.
    """

    pass
