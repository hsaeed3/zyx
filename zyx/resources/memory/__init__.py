"""zyx.resources.memory

Provides Vector Database based resources that can be used by an agent or
model to store and query memories.

NOTE:
The implementation structure of this resource is essentially a rewrite of the
`Memory` system within the `marvin` library. You can find their implementation
here:

[Marvin Memory](https://github.com/PrefectHQ/marvin/blob/main/src/marvin/memory/memory.py)
"""

from __future__ import annotations

import asyncio
import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Literal, TypeAlias

from pydantic_ai.toolsets import FunctionToolset

from ..._aliases import PydanticAITool, PydanticAIToolset
from ..abstract import AbstractResource


def sanitize_memory_key(key: str) -> str:
    # Remove any characters that are not alphanumeric or underscore
    return re.sub(r"[^a-zA-Z0-9_]", "", key)


MemoryProviderName: TypeAlias = Literal[
    "chroma/persistent",
    "chroma/ephemeral",
]


@dataclass(kw_only=True)
class MemoryProvider(ABC):
    """
    Abstract base class for a memory provider.
    """

    def configure(self, key: str) -> None:
        """
        Configure this memory provider with a given identifier or key.
        """

    @abstractmethod
    async def add(self, key: str, content: str) -> str:
        """
        Add some content to the index of memories, which will return it's
        memory id.
        """

    @abstractmethod
    async def delete(self, key: str, id: str) -> None:
        """Delete a memory by it's ID from the index of memories."""

    @abstractmethod
    async def search(
        self, key: str, query: str, n: int = 20
    ) -> Dict[str, str]:
        """Search the index for `n` memories using a string query."""


@dataclass(kw_only=True)
class Memory(AbstractResource):
    """
    A `Resource` that can be used by a model/agent to store, query and retrieve
    memories that are stored in a vector database.

    A `Memory` resource can be configured based on a given `MemoryProvider`.
    """

    key: str = field(kw_only=False)

    provider: MemoryProvider | MemoryProviderName = "chroma/persistent"
    """A classification (name) or pre-configured instance of a `MemoryProvider` that will be used
    to store and query memories."""

    instructions: str | None = field(default=None)
    """Optional instructions for the model or agent to follow, which should explain the reason
    for these memories and how it should be used."""

    auto: bool = field(default=True)
    """Whether memories are automatically included within the instructions/context of a request,
    on all requests using the message history of the context to query before the model
    or agent is invoked."""

    def __hash__(self) -> int:
        return id(self)

    def __post_init__(self) -> None:
        self.key = sanitize_memory_key(self.key)
        self._provider = get_memory_provider(self.provider)

    @property
    def provider(self) -> MemoryProvider:
        """The `MemoryProvider` instance that will be used to store and query memories."""
        return self._provider

    async def async_add(self, content: str) -> str:
        """
        Add some content to the index of memories, which will return it's
        memory id.
        """
        return await self.provider.add(self.key, content)

    def add(self, content: str) -> str:
        """Add some content to the index of memories, which will return its
        memory id."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot call synchronous 'add' inside an async context. Use 'await async_add'."
            )
        else:
            return asyncio.run(self.async_add(content))

    async def async_delete(self, id: str) -> None:
        """Delete a memory by it's ID from the index of memories."""
        return await self.provider.delete(self.key, id)

    def delete(self, id: str) -> None:
        """Delete a memory by it's ID from the index of memories."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot call synchronous 'delete' inside an async context. Use 'await async_delete'."
            )
        else:
            return asyncio.run(self.async_delete(id))

    async def async_search(self, query: str, n: int = 20) -> Dict[str, str]:
        """Search the index for `n` memories using a string query."""
        return await self.provider.search(self.key, query, n)

    def search(self, query: str, n: int = 20) -> Dict[str, str]:
        """Search the index for `n` memories using a string query."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot call synchronous 'search' inside an async context. Use 'await async_search'."
            )
        else:
            return asyncio.run(self.async_search(query, n))

    def get_toolset(self) -> PydanticAIToolset:
        write_tools = [
            PydanticAITool(
                function=self.add,
                name="add_memory",
                description="Add a memory to the index of memories.",
            ),
            PydanticAITool(
                function=self.delete,
                name="delete_memory",
                description="Delete a memory by it's ID from the index of memories.",
            ),
        ]
        read_tools = [
            PydanticAITool(
                function=self.search,
                name="search_memory",
                description="Search the index of memories for a given query.",
            ),
        ]

        if self.writeable:
            return FunctionToolset(tools=[*write_tools, *read_tools])
        else:
            return FunctionToolset(tools=read_tools)

    def get_description(self) -> str:
        if not self.instructions:
            return ""
        return f"[MEMORY : {self.key}]\n\n{self.instructions}"

    def get_state_description(self) -> str:
        return ""


def get_memory_provider(
    provider: MemoryProvider | MemoryProviderName,
) -> MemoryProvider:
    """
    Get a `MemoryProvider` instance based on a given name or pre-configured instance.
    """
    if isinstance(provider, MemoryProvider):
        return provider

    elif provider.startswith("chroma/"):
        from .chroma import ChromaMemoryProvider

        if provider == "chroma/persistent":
            return ChromaMemoryProvider(client="persistent")
        elif provider == "chroma/ephemeral":
            return ChromaMemoryProvider(client="ephemeral")

    else:
        raise ValueError(f"Invalid memory provider name: {provider}")
