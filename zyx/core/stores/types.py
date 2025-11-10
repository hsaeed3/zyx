"""zyx.core.stores.types"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Literal,
    TypeAliasType,
    TypeVar,
)

if TYPE_CHECKING:
    from ..processing.schemas.schema import Schema

__all__ = [
    "DEFAULT_STORE_TABLE_NAME",
    "StoreLocationName",
    "StoreDistanceMetric",
    "StoreSettings",
    "StoreItem",
    "StoreSearchResult",
    "TypeEntry",
]

T = TypeVar("T")

DEFAULT_STORE_TABLE_NAME = "zyx_store"
"""The default table name to use for the store."""


StoreLocationName = TypeAliasType(
    "StoreLocationName",
    Literal[
        ":memory:",
        "file::memory:?cache=shareddatabase.duckdb",
    ],
)
"""Helper alias for 'common' store location names.

NOTE: using `file::memory:?cache=shared` is recommended for in-memory storage 
to avoid issues with concurrent access, as this is an operation
that is common within the library."""


StoreDistanceMetric = TypeAliasType(
    "StoreDistanceMetric",
    Literal["cosine",],
)
"""Alias for the distance metric to use for the store. 
Currently, only cosine distance is supported.
"""


@dataclass
class TypeEntry(Generic[T]):
    """Internal representation of a type entry in the type registry.

    This holds both the factory function (or type) used to construct
    instances, and an optional Schema for validation and serialization.
    """

    factory: Any
    """The factory function or type constructor."""
    schema: "Schema[T] | None"
    """Optional schema for validation and parsing."""


@dataclass
class StoreSettings:
    """Configuration settings for a Store instance.

    This dataclass holds all the configuration parameters that define
    how a store operates, including location, embedding settings, and
    deduplication behavior.
    """

    location: str | StoreLocationName
    """The location of the store. Can be ':memory:' for in-memory or a file path."""
    embeddings: bool
    """Whether to enable embeddings for semantic search."""
    embedding_model: Any | None
    """The embedding model instance to use (if embeddings are enabled)."""
    distance_metric: StoreDistanceMetric
    """The distance metric to use for similarity calculations."""
    deduplicate: bool
    """Whether to automatically deduplicate entries by base key."""
    table_name: str
    """The name of the database table to use for storage."""


@dataclass
class StoreItem(Generic[T]):
    """A single item retrieved from a store.

    This represents a stored item with its content, metadata, and
    optionally a parsed/typed version of the content.
    """

    id: str
    """Unique identifier for the item."""
    content: str
    """The text content of the item."""
    metadata: Dict[str, Any]
    """Metadata dictionary associated with the item."""
    created_at: datetime
    """Timestamp when the item was created."""
    parsed: T | None = None
    """Optional parsed/typed version of the content."""


@dataclass
class StoreSearchResult(Generic[T]):
    """A search result from a store query.

    Contains the store item along with relevance scoring information
    from the search operation.
    """

    item: StoreItem[T]
    """The store item that matched the query."""
    score: float
    """Relevance score for this result (0.0 to 1.0, higher is better)."""
    distance: float | None = None
    """Optional distance metric value (lower is better for distance)."""
    model: str | None = None
    """The embedding model used for this search (if applicable)."""
