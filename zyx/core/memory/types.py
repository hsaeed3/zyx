"""zyx.core.memory.types"""

from __future__ import annotations

from datetime import datetime
from typing import (
    Any,
    Dict,
    Literal,
    Generic,
)
from pathlib import Path

from pydantic import BaseModel, Field
from chonkie.chunker.base import BaseChunker

from ...models.embeddings.types import EmbeddingModelName, EmbeddingModelResponse
from ...models.embeddings.model import EmbeddingModel
from ...models.adapters import ResponseModel
from ...models.language.types import (
    LanguageModelResponse,
)

__all__ = [
    "MemoryDistanceMetric",
    "MemorySettings",
    "MemoryItem",
    "MemorySearchResult",
]


MemoryDistanceMetric = Literal["cosine", "l2"]
"""Alias for the distance metrics supported by the memory
interfaces for similarity search."""


class MemorySettings(BaseModel):
    """Configuration settings for a `Memory` interface."""

    model_config = {"arbitrary_types_allowed": True}

    location: Path | str | Literal[":memory:"] = ":memory:"
    """The location or path where the memory database should be stored.
    Use `":memory:"` for an in-memory database."""

    embeddings: bool = False
    """Whether to use embeddings for storing and retrieving
    memory items. If `False`, memory items are stored as plain text."""

    embedding_model: EmbeddingModelName | str | EmbeddingModel | None = None
    """Either the name of an embedding model to infer the provider/model adapter,
    (see `zyx.models.adapters` & `zyx.models.providers`), or an initialized
    `EmbeddingModel` instance to use for generating embeddings for memory items."
    
    **WARNING**: Ensure you are aware of the embedding dimensions of the model. By default,
    this uses 1536 dimensions, to match the dimensions of OpenAI's `text-embedding-3-small` model,
    which is the default embedding model for `zyx.models.EmbeddingModel`.
    """

    dimensions: int = 1536
    """The number of dimensions the embedding model produces. Defaults to `1536` for
    compatibility with OpenAI's `text-embedding-3-small` model."""

    chunk_size: int = 2048
    """The maximum chunk size to use when generating embeddings for larger
    memory items.

    Defaults to `2048` tokens.
    """

    distance_metric: MemoryDistanceMetric = "cosine"
    """The distance metric to use when performing similarity searches
    within the memory interface.
    
    Can be either `"cosine"` or `"l2"`. Defaults to `"cosine"`.
    """

    chunker: BaseChunker | None = None
    """A custom `chonkie` chunker to use when chunking larger
    memory items for embedding generation. If `None`, uses a default
    `RecursiveChunker`."""


class MemoryItem(BaseModel):
    """Representation of a single item stored within a `Memory`
    interface.
    """

    id: str
    """A unique identifier or name for this memory item."""

    content: str
    """The main content of this memory item in string format."""

    embedding: bytes | None = Field(repr=False, default=None)
    """An embedding representation of this memory item in bytes format."""

    chunk_index: int | None = None
    """If this memory item is part of a larger chunked set,
    this represents the index of the chunk.
    """

    chunk_count: int | None = None
    """If this memory item is part of a larger chunked set,
    this represents the total number of chunks.
    """

    created_at: datetime = Field(default_factory=datetime.now)
    """The timestamp when this memory item was created."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Any additional metadata associated with this memory item."""


class MemorySearchResult(BaseModel):
    """Representation of a single search result
    returned from a `Memory` interface search.
    """

    item: MemoryItem
    """The actual memory item that was found."""

    score: float
    """The similarity score or relevance score
    associated with this search result.
    """

    distance: float | None = None
    """The distance metric (if applicable)
    associated with this search result.
    """

    model: str | None = Field(default=None)
    """If this search result was retreived using vector search,
    this is the embedding model used to generate the query
    vector."""

    @property
    def content(self) -> str:
        """Convenience property to access the content
        of the underlying memory item.
        """
        return self.item.content

    @property
    def metadata(self) -> Dict[str, Any]:
        """Convenience property to access the metadata
        of the underlying memory item.
        """
        return self.item.metadata

    @property
    def id(self) -> str:
        """Convenience property to access the ID
        of the underlying memory item.
        """
        return self.item.id

    def __str__(self) -> str:
        return _pretty_print_memory_search_result(self)

    def __rich__(self):
        return _rich_pretty_print_memory_search_result(self)


class MemoryQueryResponse(BaseModel, Generic[ResponseModel]):
    """Representation of a query result returned from querying
    a language model against the search results of a `Memory` interface.
    """

    response: LanguageModelResponse[ResponseModel]
    """The response generated by the language model
    based on the memory search results.
    """

    results: list[MemorySearchResult]
    """The list of memory search results that were
    used to generate the language model response.
    """

    @property
    def language_model(self):
        return self.response.model

    @property
    def output(self) -> ResponseModel:
        return self.response.output

    def __str__(self) -> str:
        return _pretty_print_memory_query_response(self)

    def __rich__(self):
        return _rich_pretty_print_memory_query_response(self)


def _pretty_print_memory_search_result(result: MemorySearchResult) -> str:
    content = f"MemorySearchResult: {result.id}\n"
    content += f"Score: {result.score}\n"
    if result.distance is not None:
        content += f"Distance: {result.distance}\n"
    content += (
        f"Content: {result.content[:100]}{'...' if len(result.content) > 100 else ''}\n"
    )
    return content


def _rich_pretty_print_memory_search_result(result: MemorySearchResult):
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.console import Group
    from rich.markup import escape

    table = Table(
        show_edge=False,
        show_header=True,
        expand=False,
        row_styles=["none", "dim"],
        box=box.SIMPLE,
    )
    table.add_column(
        f"\n[bold dodger_blue2]MemorySearchResult: {escape(result.id)}",
        style="bold dodger_blue2",
        no_wrap=True,
    )
    table.add_column("", justify="right")

    table.add_row("", f"[italic]{escape(str(result.score))}[/italic]")

    if result.distance is not None:
        table.add_row(
            "[dim sandy_brown]Distance:[/dim sandy_brown]",
            f"[dim italic]{escape(str(result.distance))}[/dim italic]",
        )

    content = result.content[:100] + ("..." if len(result.content) > 100 else "")
    table.add_row(
        "[dim sandy_brown]Content:[/dim sandy_brown]",
        f"[dim italic]{escape(content)}[/dim italic]",
    )
    return Group(table, Text.from_markup(""))


def _pretty_print_memory_query_response(response: MemoryQueryResponse) -> str:
    content = "MemoryQueryResponse:\n"
    content += "-------------------\n"
    content += f"{response.output}\n\n"
    content += f"Result Items: {len(response.results)}\n"
    if response.results:
        content += (
            f"Highest Score: {max(result.score for result in response.results)}\n"
        )
    content += f"Language Model: {response.language_model}\n"
    if response.results:
        if response.results[0].model:
            content += f"Query Embedding Model: {response.results[0].model}\n"
            if response.results[0].item.embedding is not None:
                content += (
                    f"Dimensions: {len(response.results[0].item.embedding) // 4}\n"
                )

        content += "\n"
    return content


def _rich_pretty_print_memory_query_response(response: MemoryQueryResponse):
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.console import Group
    from rich.markup import escape

    table = Table(
        show_edge=False,
        show_header=True,
        expand=False,
        row_styles=["none", "dim"],
        box=box.SIMPLE,
    )
    table.add_column(
        f"\n[bold dodger_blue2]MemoryQueryResponse:",
        style="bold dodger_blue2",
        no_wrap=True,
    )
    table.add_column("", justify="right")

    table.add_row("", f"[italic]{escape(str(response.output))}[/italic]\n")

    table.add_row(
        "[dim sandy_brown]Result Items:[/dim sandy_brown]",
        f"[dim italic]{len(response.results)}[/dim italic]",
    )

    if response.results:
        table.add_row(
            "[dim sandy_brown]Highest Score:[/dim sandy_brown]",
            f"[dim italic]{max(result.score for result in response.results)}[/dim italic]",
        )

    table.add_row(
        "[dim sandy_brown]Language Model:[/dim sandy_brown]",
        f"[dim italic]{escape(response.language_model)}[/dim italic]",
    )

    if response.results and response.results[0].model:
        table.add_row(
            "[dim sandy_brown]Query Embedding Model:[/dim sandy_brown]",
            f"[dim italic]{escape(response.results[0].model)}[/dim italic]",
        )
        if response.results[0].item.embedding is not None:
            dimensions = len(response.results[0].item.embedding) // 4
            table.add_row(
                "[dim sandy_brown]Dimensions:[/dim sandy_brown]",
                f"[dim italic]{dimensions}[/dim italic]",
            )

    return Group(table, Text.from_markup(""))
