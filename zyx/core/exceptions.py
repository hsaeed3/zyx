"""zyx.core.exceptions"""

from re import S
from typing import Any

from rich import get_console, traceback

traceback.install(console=get_console())

__all__ = [
    "zyxError",
    # ----- CORE -------
    "SchemaError",
    "SchemaValidationError",
    "SchemaExecutionError",
    "MemoryError",
    "MemoryStorageError",
    "MemorySearchError",
    # ----- MODELS -------
    "ModelClientError",
    "ModelDefinitionError",
    "ModelRequestError",
    # ----- UTILS -------
    "ProcessingError",
]


class zyxError(Exception):
    """Base class for all *custom* exceptions raised by the
    zyx framework.
    """

    pass


# ----- CORE -------


class SchemaError(zyxError):
    """Base class for all exceptions raised by the `zyx.core.schemas`
    module.
    """

    pass


class SchemaValidationError(SchemaError):
    """Exception raised when a schema validation fails."""

    def __init__(self, message: str, errors: list[dict[str, Any]]) -> None:
        self.message = message
        self.errors = errors
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message}"


class SchemaExecutionError(SchemaError):
    """Exception raised when a schema execution fails."""

    def __init__(
        self, message: str, errors: list[dict[str, Any]], name: str | None = None
    ) -> None:
        self.message = message
        self.errors = errors
        self.name = name
        super().__init__(message, name)

    def __str__(self) -> str:
        return f"{self.message}: {self.errors} (name: {self.name})"


class MemoryError(zyxError):
    """Base class for all exceptions raised by the `zyx.core.memory` module."""

    pass


class MemoryStorageError(MemoryError):
    """Exception raised when a memory storage operation fails."""

    def __init__(self, message: str, operation: str, key: str | None = None) -> None:
        self.message = message
        self.operation = operation
        self.key = key
        super().__init__(message)

    def __str__(self) -> str:
        key_info = f" (key: {self.key})" if self.key else ""
        return f"{self.message}\nOperation: {self.operation}{key_info}"


class MemorySearchError(MemoryError):
    """Exception raised when a memory search operation fails."""

    def __init__(
        self, message: str, query: str | None = None, reason: str | None = None
    ) -> None:
        self.message = message
        self.query = query
        self.reason = reason
        super().__init__(message)

    def __str__(self) -> str:
        query_info = f"\nQuery: {self.query}" if self.query else ""
        reason_info = f"\nReason: {self.reason}" if self.reason else ""
        return f"{self.message}{query_info}{reason_info}"


# ----- MODELS -------


class ModelClientError(zyxError):
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


class ModelDefinitionError(zyxError):
    """Exception raised when the 'definition' or initialization of a model
    and its associated components fails."""

    def __init__(self, message: str, model: str):
        self.message = message
        self.model = model
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message}\nModel: {self.model}"


class ModelRequestError(zyxError):
    """Exception raised when a model request fails."""

    def __init__(self, message: str, model: str):
        self.message = message
        self.model = model
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message}\nModel: {self.model}"


# ----- UTILS -------


class ProcessingError(zyxError):
    """Exception raised when a processing operation fails."""

    def __init__(self, message: str, kind: str):
        self.kind = kind
        self.message = message
        super().__init__(message, kind)

    def __str__(self) -> str:
        return f"{self.message}\nKind: {self.kind}"
