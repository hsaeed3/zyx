"""zyx.core.interfaces.stuff"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Type, TypeVar

from ..._internal._exceptions import StuffError
from ..processing.schemas.schema import Schema
from ..stores.type import TypeStore
from ..stores.types import (
    StoreDistanceMetric,
    StoreItem,
    StoreLocationName,
    StoreSearchResult,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from ..models.embeddings.model import (
        EmbeddingModel,
        EmbeddingModelName,
    )

__all__ = ["Stuff", "stuff"]


T = TypeVar("T")


class Stuff(Generic[T]):
    """A 'stuff' is a type-safe collection of things with semantic search capabilities.

    Stuff uses DuckDB as the storage backend, providing fast vector similarity search
    and metadata filtering. It's designed for prototyping and experimentation with
    small to medium-sized collections (< 10k items).

    Unlike a generic store, Stuff is type-locked - all items must conform to the
    specified type. This provides type safety and automatic serialization/deserialization.

    Parameters
    ----------
    type : Type[T] | Schema[T] | Dict[str, Any]
        Type specification for items in this collection. Can be:
        - A single type (e.g., `str`, `int`, `MyModel`) for all items
        - A dict mapping keys to types for different item types
        - A Schema instance for advanced configuration
    location : StoreLocationName | str | Path
        Storage location. Use ":memory:" for in-memory, or path for persistent storage.
    embeddings : bool
        Whether to generate embeddings for semantic search.
    model : EmbeddingModelName | EmbeddingModel | str
        Embedding model to use (if embeddings=True).
    distance_metric : StoreDistanceMetric
        Distance metric for similarity calculations.
    deduplicate : bool
        Whether to deduplicate items by base key on add.
    connection : DuckDBPyConnection | None
        Optional existing DuckDB connection to reuse.
    table_name : str
        Name of the table to use for storage.

    Examples
    --------
    >>> # Create an in-memory collection of strings
    >>> stuff = to_stuff(str)
    >>> stuff.add("Hello world", key="greeting")
    'greeting'

    >>> # Persistent storage
    >>> stuff = to_stuff(int, location="numbers.db")
    >>> stuff.add(42, key="answer")

    >>> # Semantic search with embeddings
    >>> stuff = to_stuff(str, embeddings=True)
    >>> stuff.add("Python is great")
    >>> results = stuff.search("programming languages")

    >>> # Type-safe with Pydantic models
    >>> from pydantic import BaseModel
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    >>>
    >>> people = to_stuff(Person)
    >>> people.add(Person(name="Alice", age=30))

    >>> # Multiple types by key
    >>> stuff = to_stuff({"text": str, "count": int})
    >>> stuff.add("hello", key="text")
    >>> stuff.add(42, key="count")
    """

    @property
    def type(self) -> Type[T] | Dict[str, Any]:
        """The type specification for this collection."""
        return self._store.type

    @property
    def schema(self) -> Schema[T] | None:
        """Schema representation if initialized with a single type."""
        return self._schema

    @property
    def store(self) -> TypeStore[T]:
        """The underlying TypeStore instance."""
        return self._store

    def __init__(
        self,
        type: Type[T] | Schema[T] | Dict[str, Any] = str,
        location: StoreLocationName | str | Path = ":memory:",
        *,
        embeddings: bool = False,
        model: "EmbeddingModelName | EmbeddingModel | str" = "openai/text-embedding-3-small",
        distance_metric: StoreDistanceMetric = "cosine",
        deduplicate: bool = True,
        connection: "DuckDBPyConnection | None" = None,
        table_name: str = "stuff",
    ) -> None:
        """Initialize a Stuff collection with the given type and configuration."""
        # Store schema if provided
        if isinstance(type, Schema):
            self._schema = type
            type_spec = type.source
        else:
            # Try to create schema for single types
            if not isinstance(type, dict):
                self._schema = Schema(type)
                type_spec = type
            else:
                self._schema = None
                type_spec = type

        # Initialize the underlying TypeStore
        self._store = TypeStore(
            type=type_spec,
            location=location,
            embeddings=embeddings,
            model=model,
            distance_metric=distance_metric,
            deduplicate=deduplicate,
            connection=connection,
            table_name=table_name,
        )

    def add(
        self,
        content: T | Any,
        key: str | None = None,
        *,
        metadata: Dict[str, Any] | None = None,
        embed: bool = True,
    ) -> str:
        """Add an item to the collection.

        Parameters
        ----------
        content : T | Any
            The content to add. Must match the collection's type.
        key : str | None
            Optional custom key/ID for the item. Auto-generated if None.
        metadata : Dict[str, Any] | None
            Optional metadata to associate with the item.
        embed : bool
            Whether to generate embeddings (only if embeddings are enabled).

        Returns
        -------
        str
            The key of the added item.

        Raises
        ------
        StuffError
            If the content type doesn't match the collection type.

        Examples
        --------
        >>> stuff = to_stuff(str)
        >>> stuff.add("hello", key="greeting")
        'greeting'

        >>> # Auto-generated key
        >>> key = stuff.add("world")
        >>> print(key)
        'a1b2c3...'

        >>> # With metadata
        >>> stuff.add("important", metadata={"priority": "high"})
        """
        try:
            return self._store.add(
                content=content,
                key=key,
                metadata=metadata,
                embed=embed,
            )
        except Exception as e:
            raise StuffError(f"Failed to add item: {e}") from e

    def get(self, key: str) -> StoreItem[T] | None:
        """Get an item by its key.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.

        Returns
        -------
        StoreItem[T] | None
            The store item if found, None otherwise.

        Examples
        --------
        >>> stuff = to_stuff(str)
        >>> stuff.add("hello", key="greeting")
        >>> item = stuff.get("greeting")
        >>> print(item.parsed)
        'hello'
        """
        return self._store.get(key)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
    ) -> List[StoreSearchResult[T]]:
        """Search the collection for similar items.

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int
            Number of results to return.
        filter : Dict[str, Any] | None
            Optional metadata filter.
        hybrid : bool
            Use hybrid search (vector + keyword). Ignored if embeddings=False.
        hybrid_weight : float
            Weight for vector search (0=keyword only, 1=vector only).

        Returns
        -------
        List[StoreSearchResult[T]]
            List of search results sorted by relevance.

        Examples
        --------
        >>> stuff = to_stuff(str, embeddings=True)
        >>> stuff.add("Python is great")
        >>> stuff.add("Java is powerful")
        >>> results = stuff.search("programming languages", top_k=2)
        >>> print(results[0].item.parsed)
        'Python is great'
        """
        return self._store.search(
            query=query,
            top_k=top_k,
            filter=filter,
            hybrid=hybrid,
            hybrid_weight=hybrid_weight,
        )

    def list(
        self,
        query: str | None = None,
        *,
        keys: bool = False,
        strategy: str = "semantic",
        top_k: int | None = None,
        filter: Dict[str, Any] | None = None,
    ) -> List[T] | List[Dict[str, Any]]:
        """List items in the collection, optionally with search.

        Parameters
        ----------
        query : str | None
            Optional query to filter/search items.
        keys : bool
            If True, returns dicts with keys and values. If False, returns just values.
        strategy : str
            Search strategy: "semantic", "bm25", or "hybrid".
        top_k : int | None
            Number of results to return. None returns all items.
        filter : Dict[str, Any] | None
            Optional metadata filter.

        Returns
        -------
        List[T] | List[Dict[str, Any]]
            List of items or key-value dicts.

        Examples
        --------
        >>> stuff = to_stuff(int)
        >>> stuff.add(42, key="answer")
        >>> stuff.add(13, key="lucky")
        >>>
        >>> # List all values
        >>> print(stuff.list())
        [42, 13]
        >>>
        >>> # List with keys
        >>> print(stuff.list(keys=True))
        [{'key': 'answer', 'value': 42}, {'key': 'lucky', 'value': 13}]
        >>>
        >>> # Search with query
        >>> stuff = to_stuff(str, embeddings=True)
        >>> stuff.add("Python programming")
        >>> stuff.add("Java development")
        >>> print(stuff.list("Python"))
        ['Python programming']
        """
        if query:
            # Search mode
            if strategy == "semantic":
                results = self.search(
                    query=query,
                    top_k=top_k or 100,
                    filter=filter,
                    hybrid=False,
                    hybrid_weight=1.0,
                )
            elif strategy == "bm25":
                results = self.search(
                    query=query,
                    top_k=top_k or 100,
                    filter=filter,
                    hybrid=False,
                    hybrid_weight=0.0,
                )
            else:  # hybrid
                results = self.search(
                    query=query,
                    top_k=top_k or 100,
                    filter=filter,
                    hybrid=True,
                )

            if keys:
                return [
                    {"key": result.item.id, "value": result.item.parsed}
                    for result in results
                ]
            else:
                return [result.item.parsed for result in results]
        else:
            # List all mode - we'll need to query the store directly
            # For now, use empty search which returns everything
            results = self.search(
                query="",
                top_k=top_k or 10000,
                filter=filter,
                hybrid=False,
            )

            if keys:
                return [
                    {"key": result.item.id, "value": result.item.parsed}
                    for result in results
                ]
            else:
                return [result.item.parsed for result in results]

    def delete(self, key: str) -> bool:
        """Delete an item and all its chunks by key.

        Parameters
        ----------
        key : str
            The key of the item to delete.

        Returns
        -------
        bool
            True if the item was deleted, False otherwise.

        Examples
        --------
        >>> stuff = to_stuff(str)
        >>> stuff.add("hello", key="greeting")
        >>> stuff.delete("greeting")
        True
        >>> stuff.get("greeting")
        None
        """
        return self._store.delete(key)

    def clear(self) -> None:
        """Clear all items from the collection.

        Examples
        --------
        >>> stuff = to_stuff(str)
        >>> stuff.add("item1")
        >>> stuff.add("item2")
        >>> stuff.count()
        2
        >>> stuff.clear()
        >>> stuff.count()
        0
        """
        self._store.clear()

    def count(self) -> int:
        """Get the total number of unique items in the collection.

        Returns
        -------
        int
            Number of unique items (by base key).

        Examples
        --------
        >>> stuff = to_stuff(str)
        >>> stuff.add("item1")
        >>> stuff.add("item2")
        >>> stuff.count()
        2
        """
        return self._store.count()

    def close(self) -> None:
        """Close the database connection.

        Examples
        --------
        >>> stuff = to_stuff(str, location="data.db")
        >>> stuff.add("content")
        >>> stuff.close()
        """
        self._store.close()

    def __enter__(self) -> "Stuff[T]":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __len__(self) -> int:
        """Return the number of items in the collection."""
        return self.count()

    def __repr__(self) -> str:
        """String representation of the collection."""
        type_info = (
            self.type.__name__
            if hasattr(self.type, "__name__")
            else str(self.type)
        )
        return (
            f"Stuff(type={type_info}, "
            f"location={self._store.settings.location!r}, "
            f"count={self.count()})"
        )


def to_stuff(
    type: Type[T] | Schema[T] | Dict[str, Any] = str,
    location: StoreLocationName | str | Path = ":memory:",
    *,
    embeddings: bool = True,
    model: "EmbeddingModelName | EmbeddingModel | str" = "openai/text-embedding-3-small",
    distance_metric: StoreDistanceMetric = "cosine",
    deduplicate: bool = True,
    connection: "DuckDBPyConnection | None" = None,
    table_name: str = "stuff",
) -> Stuff[T]:
    """Create a new Stuff collection with the given type and configuration.

    A Stuff is a type-safe collection of things with semantic search capabilities.

    Parameters
    ----------
    type : Type[T] | Schema[T] | Dict[str, Any]
        Type specification for items in this collection.
    location : StoreLocationName | str | Path
        Storage location (":memory:" for in-memory or path for persistent).
    embeddings : bool
        Whether to generate embeddings for semantic search.
    model : EmbeddingModelName | EmbeddingModel | str
        Embedding model to use (if embeddings=True).
    distance_metric : StoreDistanceMetric
        Distance metric for similarity calculations.
    deduplicate : bool
        Whether to deduplicate items by base key on add.
    connection : DuckDBPyConnection | None
        Optional existing DuckDB connection to reuse.
    table_name : str
        Name of the table to use for storage.

    Returns
    -------
    Stuff[T]
        A new Stuff collection.

    Examples
    --------
    >>> # In-memory collection of strings
    >>> stuff = to_stuff(str)
    >>> stuff.add("Hello")

    >>> # Persistent storage with embeddings
    >>> stuff = to_stuff(str, location="data.db", embeddings=True)
    >>> stuff.add("Python is great")
    >>> results = stuff.search("programming")

    >>> # Type-safe with Pydantic models
    >>> from pydantic import BaseModel
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    >>>
    >>> people = to_stuff(Person, embeddings=False)
    >>> people.add(Person(name="Alice", age=30))

    >>> # Multiple types by key
    >>> stuff = to_stuff({"text": str, "count": int})
    >>> stuff.add("hello", key="text")
    >>> stuff.add(42, key="count")
    """
    return Stuff(
        type=type,
        location=location,
        embeddings=embeddings,
        model=model,
        distance_metric=distance_metric,
        deduplicate=deduplicate,
        connection=connection,
        table_name=table_name,
    )
