"""zyx.core.stores.type"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Type, TypeVar

from .store import Store
from .types import (
    DEFAULT_STORE_TABLE_NAME,
    StoreDistanceMetric,
    StoreItem,
    StoreLocationName,
    StoreSearchResult,
    TypeEntry,
)
from .utils import json_ready

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from ..models.embeddings.model import (
        EmbeddingModel,
        EmbeddingModelName,
    )
    from ..processing.schemas.schema import Schema

__all__ = ["TypeStore"]

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class TypeStore(Store, Generic[T]):
    """Type-safe store with automatic serialization/deserialization.

    `TypeStore` extends the base `Store` with type-aware storage capabilities.
    It can automatically serialize and deserialize values according to their
    type specifications, supporting:

    - Single type for all entries (e.g., `str`, `int`, `MyModel`)
    - Multiple types mapped by key (e.g., `{"user": User, "post": Post}`)
    - Schema-based validation and parsing
    - Default type fallback with `__default__` key

    Parameters
    ----------
    type : Type[T] | Dict[str, Any]
        Type specification. Can be:
        - A single type (e.g., `str`, `int`, `MyModel`) for all keys
        - A dict mapping keys to types (e.g., `{"person": Person, "age": int}`)
        - A dict with `"__default__"` key for fallback type
    location : StoreLocationName | str | Path
        Path to the database file or ':memory:' for in-memory storage.
    embeddings : bool
        Whether to enable embedding generation for semantic search.
    model : EmbeddingModelName | EmbeddingModel | str
        Embedding model to use (if embeddings=True).
    distance_metric : StoreDistanceMetric
        Distance metric for similarity calculations (default: 'cosine').
    deduplicate : bool
        Whether to deduplicate items by base key on add.
    connection : DuckDBPyConnection | None
        Optional existing DuckDB connection to reuse.
    table_name : str
        Name of the table to use for storage.

    Examples
    --------
    ```python
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Person:
        ...     name: str
        ...     age: int

        >>> # Single type for all entries
        >>> store = TypeStore(Person)
        >>> store.add(Person("Alice", 30), key="alice")
        'alice'
        >>> item = store.get("alice")
        >>> print(item.parsed)
        Person(name='Alice', age=30)

        >>> # Multiple types by key
        >>> store = TypeStore({"person": Person, "count": int})
        >>> store.add(Person("Bob", 25), key="person")
        >>> store.add(42, key="count")
        >>> person = store.get("person")
        >>> print(person.parsed)
        Person(name='Bob', age=25)

        >>> # With default fallback
        >>> store = TypeStore({"__default__": str, "person": Person})
        >>> store.add("some text", key="text")  # Uses default str type
        >>> store.add(Person("Charlie", 35), key="person")  # Uses Person type
    ```
    """

    _type_map: Dict[str, TypeEntry]
    _default_type_entry: TypeEntry
    _type_spec: Type[T] | Dict[str, Any]

    @property
    def type(self) -> Type[T] | Dict[str, Any]:
        """Get the type specification for this store.

        Returns the original type specification passed during initialization.
        """
        return self._type_spec

    def __init__(
        self,
        type: Type[T] | Dict[str, Any],
        location: StoreLocationName | str | Path = ":memory:",
        *,
        embeddings: bool = True,
        model: "EmbeddingModelName | EmbeddingModel | str" = "openai/text-embedding-3-small",
        distance_metric: StoreDistanceMetric = "cosine",
        deduplicate: bool = True,
        connection: "DuckDBPyConnection | None" = None,
        table_name: str = DEFAULT_STORE_TABLE_NAME,
    ) -> None:
        """Initialize the TypeStore instance."""
        # Store the original type specification
        self._type_spec = type

        # Normalize and store type specifications
        self._type_map, self._default_type_entry = (
            self._normalize_type_spec(type)
        )

        # Initialize the base Store
        super().__init__(
            location=location,
            embeddings=embeddings,
            model=model,
            distance_metric=distance_metric,
            deduplicate=deduplicate,
            connection=connection,
            table_name=table_name,
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Type management
    # ─────────────────────────────────────────────────────────────────────────────

    def _normalize_type_spec(
        self, spec: Dict[str, Any] | Any
    ) -> tuple[Dict[str, TypeEntry], TypeEntry]:
        """Normalize type specification into internal format.

        Parameters
        ----------
        spec : Dict[str, Any] | Any
            Type specification. Can be a single type or a dict of key->type mappings.

        Returns
        -------
        tuple[Dict[str, TypeEntry], TypeEntry]
            (type_map, default_type_entry)
        """
        if isinstance(spec, dict):
            default = spec.get("__default__", str)
            mapping = {
                key: self._build_type_entry(value)
                for key, value in spec.items()
                if key != "__default__"
            }
            return mapping, self._build_type_entry(default)
        return {}, self._build_type_entry(spec)

    @lru_cache(maxsize=128)
    def _build_type_entry(self, value: Any) -> TypeEntry:
        """Build a type entry from a value.

        Parameters
        ----------
        value : Any
            The type or schema to build an entry from.

        Returns
        -------
        TypeEntry
            An internal type entry containing factory and optional schema.

        Note: Cached for performance as type entry building can be expensive.
        """
        from ..processing.schemas.schema import Schema

        if isinstance(value, Schema):
            return TypeEntry(factory=value, schema=value)
        if value is str:
            return TypeEntry(factory=str, schema=None)
        try:
            schema = Schema(value)
        except Exception:
            schema = None
        return TypeEntry(factory=value, schema=schema)

    # ─────────────────────────────────────────────────────────────────────────────
    # Override serialization/deserialization
    # ─────────────────────────────────────────────────────────────────────────────

    def _serialize_value(
        self, base_key: str, value: Any
    ) -> tuple[str, str]:
        """Serialize a value according to its type entry.

        Parameters
        ----------
        base_key : str
            The base key to look up the type specification.
        value : Any
            The value to serialize.

        Returns
        -------
        tuple[str, str]
            (text_content, json_payload)

        Raises
        ------
        ValueError
            If the value does not conform to the expected schema.
        """
        # Optimized: Use local variable for method lookup
        entry = self._type_map.get(base_key, self._default_type_entry)

        # Validate with schema if available
        if entry.schema:
            try:
                value = entry.schema(value)
            except Exception as exc:
                raise ValueError(
                    f"Value does not conform to schema for key '{base_key}'."
                ) from exc

        json_ready_value = json_ready(value)
        payload = self._serialize_payload(json_ready_value)

        # For text content, use stable string from utils
        # Optimized: Import at module level would be better, but keep local for now
        from .utils import stable_string

        text_value = stable_string(json_ready_value)

        return text_value, payload

    def _deserialize_payload(
        self, base_key: str, payload: str | None
    ) -> T | Any:
        """Deserialize a payload according to its type entry.

        Parameters
        ----------
        base_key : str
            The base key to look up the type specification.
        payload : str | None
            The JSON payload to deserialize.

        Returns
        -------
        T | Any
            The deserialized value according to the type specification.
        """
        if payload is None:
            return None

        # Optimized: More efficient JSON parsing with early exit
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = payload

        # Optimized: Use local variable
        entry = self._type_map.get(base_key, self._default_type_entry)

        # Try schema validation/parsing first
        if entry.schema:
            try:
                return entry.schema(data)
            except Exception:
                pass

        # Fall back to factory function
        factory = entry.factory
        # Optimized: Faster type check
        if factory is None or factory is str:
            return data if isinstance(data, str) else str(data)
        try:
            return factory(data)
        except Exception:
            return data

    # ─────────────────────────────────────────────────────────────────────────────
    # Override methods that need type-aware behavior
    # ─────────────────────────────────────────────────────────────────────────────

    def _build_search_result(
        self,
        row: Any,
        score: float,
        distance: float | None,
        model_name: str | None,
    ) -> StoreSearchResult[T]:
        """Build a StoreSearchResult with typed parsing."""
        (
            item_id,
            content,
            payload,
            metadata_json,
            created_at_raw,
        ) = row[:5]

        metadata = self._metadata_from_json(metadata_json)
        parsed_value = self._deserialize_payload(
            self._base_key(item_id), payload
        )

        item: StoreItem[T] = StoreItem(
            id=item_id,
            content=content,
            metadata=metadata,
            created_at=self._coerce_datetime(created_at_raw),
            parsed=parsed_value,
        )

        return StoreSearchResult(
            item=item,
            score=score,
            distance=distance,
            model=model_name,
        )

    async def async_add(
        self,
        content: T | Any,
        key: str | None = None,
        *,
        metadata: Dict[str, Any] | None = None,
        embed: bool = True,
    ) -> str:
        """Asynchronously add typed content to the store.

        Parameters
        ----------
        content : T | Any
            The typed content to add.
        key : str | None
            Optional custom key/ID for the item.
        metadata : Dict[str, Any] | None
            Optional metadata dictionary.
        embed : bool
            Whether to generate embeddings.

        Returns
        -------
        str
            The key of the added item.
        """
        from .utils import generate_content_key

        base_key = self._base_key(key) if key else None

        if key is None:
            key = generate_content_key(content, metadata)
            base_key = key
        elif base_key is None:
            base_key = self._base_key(key)

        # Use type-aware serialization
        text_value, payload = self._serialize_value(base_key, content)
        metadata_json = json.dumps(
            metadata or {}, default=str, ensure_ascii=False
        )

        embedding_vector: List[float] | None = None
        if self._settings.embeddings and embed:
            embedding_vector = await self._embed_text(text_value)

        if self._settings.deduplicate:
            self._delete_by_base_key(base_key)

        from datetime import datetime

        self._connection.execute(
            f"""
            INSERT INTO {self._table_name} (id, base_key, content, payload, metadata, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                key,
                base_key,
                text_value,
                payload,
                metadata_json,
                embedding_vector,
                datetime.utcnow(),
            ],
        )

        return key

    async def async_get(self, key: str) -> StoreItem[T] | None:
        """Asynchronously get a typed item by key.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.

        Returns
        -------
        StoreItem[T] | None
            The typed store item if found, None otherwise.
        """
        row = self._connection.execute(
            f"""
            SELECT id, content, payload, metadata, created_at
            FROM {self._table_name}
            WHERE id = ?
        """,
            [key],
        ).fetchone()

        if not row:
            return None

        metadata = self._metadata_from_json(row[3])
        parsed = self._deserialize_payload(self._base_key(row[0]), row[2])

        return StoreItem(
            id=row[0],
            content=row[1],
            metadata=metadata,
            created_at=self._coerce_datetime(row[4]),
            parsed=parsed,
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Sync wrappers (inherit most from Store, but override for typing)
    # ─────────────────────────────────────────────────────────────────────────────

    def add(
        self,
        content: T | Any,
        key: str | None = None,
        *,
        metadata: Dict[str, Any] | None = None,
        embed: bool = True,
    ) -> str:
        """Add typed content to the store.

        Parameters
        ----------
        content : T | Any
            The typed content to add.
        key : str | None
            Optional custom key/ID for the item.
        metadata : Dict[str, Any] | None
            Optional metadata dictionary.
        embed : bool
            Whether to generate embeddings.

        Returns
        -------
        str
            The key of the added item.

        Examples
        --------
        ```python
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class User:
            ...     name: str
            ...     id: int
            >>> store = TypeStore(types=User)
            >>> store.add(User("Alice", 1), key="user1")
            'user1'
        ```
        """
        return self._run_async(
            self.async_add(
                content, key=key, metadata=metadata, embed=embed
            )
        )

    def get(self, key: str) -> StoreItem[T] | None:
        """Get a typed item by its key.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.

        Returns
        -------
        StoreItem[T] | None
            The typed store item if found, None otherwise.

        Examples
        --------
        ```python
            >>> store = TypeStore(types=int)
            >>> store.add(42, key="answer")
            'answer'
            >>> item = store.get("answer")
            >>> print(item.parsed)
            42
        ```
        """
        return self._run_async(self.async_get(key))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
    ) -> List[StoreSearchResult[T]]:
        """Search the store for similar typed content.

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int
            Number of results to return.
        filter : Dict[str, Any] | None
            Optional metadata filter.
        hybrid : bool
            Use hybrid search (vector + keyword).
        hybrid_weight : float
            Weight for vector search (0=keyword only, 1=vector only).

        Returns
        -------
        List[StoreSearchResult[T]]
            List of typed search results sorted by relevance.

        Examples
        --------
        ```python
            >>> store = TypeStore(types=Person, embeddings=True)
            >>> store.add(Person("Alice", 30))
            >>> results = store.search("Who is Alice?")
            >>> print(results[0].item.parsed)
            Person(name='Alice', age=30)
        ```
        """
        return self._run_async(
            self.async_search(
                query,
                top_k=top_k,
                filter=filter,
                hybrid=hybrid,
                hybrid_weight=hybrid_weight,
            )
        )

    def __repr__(self) -> str:
        """String representation of the TypeStore."""
        if self._type_map:
            type_info = (
                f"{{{', '.join(repr(k) for k in self._type_map.keys())}}}"
            )
        else:
            factory = self._default_type_entry.factory
            type_name = getattr(factory, "__name__", str(factory))
            type_info = type_name
        return (
            f"TypeStore({type_info}, "
            f"location={self._settings.location!r}, "
            f"embeddings={self._settings.embeddings})"
        )
