"""zyx.core.memory.types"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generic, List, Literal, TypeVar

from ...models.language.model import LanguageModelResponse

__all__ = [
    "MemoryDistanceMetric",
    "MemorySettings",
    "MemoryItem",
    "MemorySearchResult",
    "MemoryQueryResponse",
]


T = TypeVar("T")


MemoryDistanceMetric = Literal["cosine"]


@dataclass
class MemorySettings:
    """Configuration settings for a Memory instance."""

    location: str
    embeddings: bool
    embedding_model: Any | None
    distance_metric: MemoryDistanceMetric
    deduplicate: bool
    types: Any
    table_name: str


@dataclass
class MemoryItem:
    """A single memory item retrieved from storage."""

    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    parsed: Any | None = None

    def __str__(self) -> str:
        from ..._lib._beautification import _pretty_print_memory_item

        return _pretty_print_memory_item(self)

    def __rich__(self):
        from ..._lib._beautification import _rich_pretty_print_memory_item

        return _rich_pretty_print_memory_item(self)


@dataclass
class MemorySearchResult:
    """A search result with relevance score."""

    item: MemoryItem
    score: float
    distance: float | None = None
    model: str | None = None

    def __str__(self) -> str:
        from ..._lib._beautification import _pretty_print_memory_search_result

        return _pretty_print_memory_search_result(self)

    def __rich__(self):
        from ..._lib._beautification import _rich_pretty_print_memory_search_result

        return _rich_pretty_print_memory_search_result(self)


@dataclass
class MemoryQueryResponse(Generic[T]):
    """Response from a RAG query combining search results and LLM response."""

    response: LanguageModelResponse[T]
    results: List[MemorySearchResult]

    @property
    def language_model(self) -> str:
        return self.response.model

    @property
    def output(self) -> T:
        return self.response.content

    def __str__(self) -> str:
        from ..._lib._beautification import _pretty_print_memory_query_response

        return _pretty_print_memory_query_response(self)

    def __rich__(self):
        from ..._lib._beautification import _rich_pretty_print_memory_query_response

        return _rich_pretty_print_memory_query_response(self)
