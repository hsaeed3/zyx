"""zyx.resources.abstract"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .._aliases import PydanticAIToolset


@dataclass(init=False)
class AbstractResource(ABC):
    """
    Abstract base class for all `Resources`, that can be used within the
    `tools` or `attachments` parameters of a semantic operation.

    A resource is, in literal terms, a pythonic or external resource that can
    be ... by an agent or model:

    1. Read / Interpreted
    2. Queried
    3. Written to / Mutated

    NOTE: If a resource is given within `tools`, a model will only recieve the toolset
    for that resource, if it is given within `attachments`, a model will recieve
    both the toolset and a descritpion/state representation of this resource (if applicable).
    """

    def __init__(
        self,
        name: str,
        *,
        writeable: bool = False,
        confirm: bool = True,
    ) -> None:

        self.name = name
        """Human-readable resource name."""

        self.writeable = writeable
        """Whether this resource can be mutated/written to when executing
        a semantic operation."""

        self.confirm = confirm
        """Whether mutating actions require confirmation."""

    @abstractmethod
    def get_description(self) -> str:
        """
        Returns a human-readable description of this resource.
        """

    @abstractmethod
    def get_state_description(self) -> str:
        """
        Returns a human-readable representation of the current state of this resource.

        For example, a file resource might return the contents of the file.
        """

    @abstractmethod
    def get_toolset(self) -> PydanticAIToolset:
        """
        Returns a `pydantic_ai.FunctionToolset` object that can be used by an agent to interact with this resource.
        with this resource, based on if it is writeable or not.
        """
