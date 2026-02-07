"""zyx.exceptions"""

from __future__ import annotations

from typing import Any

from pydantic_ai.agent import Agent

__all__ = (
    "ZYXError",
    "InvalidTargetError",
    "InvalidModelError",
    "AgentRunError",
)


class ZYXError(Exception):
    """Base exception for all ZYX errors."""

    pass


class InvalidTargetError(ZYXError):
    """
    Raised when a semantic operation recieves an invalid target type/object.
    """

    def __init__(
        self,
        target: type,
    ) -> None:
        super().__init__(
            (
                f"Recieved an invalid target type/object to generate a result for.\n"
                f"The target type/object {target.__name__} is not yet supported."
            )
        )


class InvalidModelError(ZYXError):
    """
    Raised when the `model` parameter is not provided or is invalid.
    """

    def __init__(
        self,
        model: Any | None = None,
    ) -> None:
        if not model:
            super().__init__(
                "Did not recieve a valid model to create an agent from.\n"
                "A model can be provided as a string, a `pydantic_ai.Model` object, or a `pydantic_ai.Agent` object."
            )

        else:
            super().__init__(
                f"Recieved an invalid input object to create/prepare an agent from.\n"
                f"Expected a string, a `pydantic_ai.Model` object, or a `pydantic_ai.Agent` object, received: {type(model)}"
            )


class AgentRunError(ZYXError):
    """
    Raised when an error occurs while running a Pydantic AI agent.
    """

    def __init__(
        self,
        operation_kind: str,
        agent: Agent,
        error: Exception,
    ) -> None:
        super().__init__(
            f"An error occurred while running the Pydantic AI agent for semantic operation: {operation_kind}\n"
            f"Agent: {agent}\n"
            f"Error: {error}"
        )
