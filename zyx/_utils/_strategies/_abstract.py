"""zyx._utils._strategies._abstract"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..._aliases import PydanticAIToolset

__all__ = ("AbstractStrategy",)


class AbstractStrategy(ABC):
    """
    Base class for 'strategies' that are provided to agents or models
    to interface with various objects and resources.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        raise NotImplementedError

    def get_description(self) -> str:
        return self.kind

    def get_state_description(self) -> str | None:
        return None

    def get_toolset(self) -> PydanticAIToolset | None:
        return None
