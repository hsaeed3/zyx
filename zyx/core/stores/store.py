"""zyx.core.stores.store"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal

from .types import (
    DEFAULT_STORE_TABLE_NAME,
    StoreDistanceMetric,
    StoreItem,
    StoreLocationName,
    StoreSearchResult,
    StoreSettings,
)
from .utils import generate_content_key, json_ready, stable_string

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from ..models.embeddings.model import (
        EmbeddingModel,
        EmbeddingModelName,
    )

__all__ = ["Store"]

_logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_duckdb():
    """Lazily load DuckDB with helpful error message."""
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "Could not import duckdb package. "
            "Please install it with `pip install duckdb`."
        ) from exc
    return duckdb


class Store:
    """Base store implementation using DuckDB for storage with optional embeddings.

    A `Store` provides a simple interface for storing, retrieving, and searching content
    with optional semantic search capabilities through embeddings. It uses DuckDB as the
    underlying storage engine, supporting both in-memory and persistent storage.

    Features:
    - Synchronous and asynchronous methods for all operations
    - Optional embedding generation for semantic search
    - Keyword search using DuckDB's string functions
    - Hybrid search combining vector and keyword signals
    - Automatic deduplication by content key
    - Metadata filtering

    Parameters
    ----------
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
        >>> # Create an in-memory store with embeddings
        >>> store = Store(embeddings=True)
        >>> store.add("Python is a programming language", key="python")
        'python'

        >>> # Search for similar content
        >>> results = store.search("What is Python?", top_k=3)
        >>> print(results[0].item.content)
        'Python is a programming language'

        >>> # Persistent storage
        >>> store = Store("my_data.duckdb", embeddings=True)
        >>> store.add("Machine learning is a subset of AI")

        >>> # Retrieve by key
        >>> item = store.get("python")
        >>> print(item.content)
        'Python is a programming language'
    ```
    """

    _connection: "DuckDBPyConnection | None" = None

    @property
    def settings(self) -> StoreSettings:
        """Store configuration settings."""
        return self._settings

    def __init__(
        self,
        location: StoreLocationName
        | str
        | Path = "file::memory:?cache=shared",
        *,
        embeddings: bool = True,
        model: "EmbeddingModelName | EmbeddingModel | str" = "openai/text-embedding-3-small",
        distance_metric: StoreDistanceMetric = "cosine",
        deduplicate: bool = True,
        connection: "DuckDBPyConnection | None" = None,
        table_name: str = DEFAULT_STORE_TABLE_NAME,
    ) -> None:
        """Initialize the Store instance."""
        from ..models.embeddings.model import EmbeddingModel

        self._duckdb = _get_duckdb()

        if embeddings:
            if model is None:
                model = "openai/text-embedding-3-small"
                _logger.debug(
                    "Embeddings enabled but no model provided, defaulting to %s",
                    model,
                )

            if isinstance(model, EmbeddingModel):
                resolved_model = model
            else:
                resolved_model = EmbeddingModel(model)
                _logger.debug("Initialized embedding model: %s", model)
        else:
            resolved_model = None
            _logger.debug("Store initialized without embeddings")

        location_str = str(location)
        _logger.debug(
            "Store location: %s, table: %s", location_str, table_name
        )

        self._settings = StoreSettings(
            location=location_str,
            embeddings=embeddings,
            embedding_model=resolved_model,
            distance_metric=distance_metric,
            deduplicate=deduplicate,
            table_name=table_name,
        )

        self._connection = connection or self._duckdb.connect(
            database=location_str,
            config={
                "enable_external_access": "false",
                "autoload_known_extensions": "false",
                "autoinstall_known_extensions": "false",
                "threads": "4",  # Enable multi-threading
            },
        )
        self._table_name = table_name
        self._ensure_table()
        self._optimize_connection()

    # ─────────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────────

    def _optimize_connection(self) -> None:
        """Optimize DuckDB connection settings for better performance."""
        try:
            # Enable parallel query execution
            self._connection.execute("SET threads TO 4")
            # Optimize for analytical queries
            self._connection.execute(
                "SET preserve_insertion_order = false"
            )
            # Increase default block size for better I/O performance
            self._connection.execute("SET default_block_size = 262144")
        except Exception as e:
            _logger.debug("Could not apply all optimizations: %s", e)

    def _ensure_table(self) -> None:
        """Ensure the store table exists with proper schema."""
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id TEXT PRIMARY KEY,
                base_key TEXT,
                content TEXT,
                payload TEXT,
                metadata TEXT,
                embedding FLOAT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self._connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_base_key
            ON {self._table_name}(base_key)
        """
        )

    @lru_cache(maxsize=2048)
    def _base_key(self, key: str) -> str:
        """Extract base key from a full key (handles chunked keys).

        Keys can have formats like 'base:chunk' or 'base::chunk'.
        This extracts the 'base' part.

        Cached for performance as this is called frequently.
        """
        if "::" in key:
            return key.split("::", 1)[0]
        if ":" in key:
            return key.split(":", 1)[0]
        return key

    def _delete_by_base_key(self, base_key: str) -> None:
        """Delete all entries with the given base key."""
        self._connection.execute(
            f"DELETE FROM {self._table_name} WHERE base_key = ?",
            [base_key],
        )

    def _serialize_value(self, value: Any) -> tuple[str, str]:
        """Serialize a value for storage.

        Returns
        -------
        tuple[str, str]
            (text_content, json_payload)
        """
        json_value = json_ready(value)
        payload = self._serialize_payload(json_value)
        text_content = stable_string(json_value)
        return text_content, payload

    def _serialize_payload(self, value: Any) -> str:
        """Serialize a value to JSON string."""
        try:
            return json.dumps(value, default=str, ensure_ascii=False)
        except TypeError:  # pragma: no cover
            return json.dumps(str(value), ensure_ascii=False)

    @lru_cache(maxsize=512)
    def _metadata_from_json(
        self, metadata_json: str | None
    ) -> Dict[str, Any]:
        """Parse metadata from JSON string.

        Cached for performance as metadata parsing can be expensive.
        Note: Returns dict, so cached return is immutable view.
        """
        if not metadata_json:
            return {}
        try:
            value = json.loads(metadata_json)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _matches_filter(
        self,
        metadata: Dict[str, Any],
        filter_values: Dict[str, Any] | None,
    ) -> bool:
        """Check if metadata matches filter criteria."""
        if not filter_values:
            return True
        return all(
            metadata.get(key) == value
            for key, value in filter_values.items()
        )

    def _coerce_datetime(self, value: Any) -> datetime:
        """Coerce a value to datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.utcnow()

    def _deduplicate_results(
        self, results: List[StoreSearchResult]
    ) -> List[StoreSearchResult]:
        """Deduplicate search results by base key, keeping highest score."""
        if not self._settings.deduplicate:
            return results

        # Optimized: Use local variable and simpler logic
        deduped: Dict[str, StoreSearchResult] = {}
        base_key_fn = self._base_key

        for result in results:
            bkey = base_key_fn(result.item.id)
            existing = deduped.get(bkey)
            # Optimized: Simpler conditional
            if existing is None or result.score > existing.score:
                deduped[bkey] = result

        return list(deduped.values())

    def _build_search_result(
        self,
        row: Any,
        score: float,
        distance: float | None,
        model_name: str | None,
    ) -> StoreSearchResult:
        """Build a StoreSearchResult from a database row."""
        (
            item_id,
            content,
            payload,
            metadata_json,
            created_at_raw,
        ) = row[:5]

        metadata = self._metadata_from_json(metadata_json)

        item = StoreItem(
            id=item_id,
            content=content,
            metadata=metadata,
            created_at=self._coerce_datetime(created_at_raw),
            parsed=payload,  # For base Store, parsed is just the raw payload
        )

        return StoreSearchResult(
            item=item,
            score=score,
            distance=distance,
            model=model_name,
        )

    async def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self._settings.embedding_model:
            raise ValueError("Embedding model is not configured.")
        response = await self._settings.embedding_model.arun(text)
        return response.data[0].embedding  # type: ignore[index]

    def _apply_filter(
        self,
        results: List[StoreSearchResult],
        filter_values: Dict[str, Any] | None,
    ) -> List[StoreSearchResult]:
        """Apply metadata filter to search results."""
        if not filter_values:
            return results
        return [
            result
            for result in results
            if self._matches_filter(result.item.metadata, filter_values)
        ]

    # ─────────────────────────────────────────────────────────────────────────────
    # Public API (sync wrappers)
    # ─────────────────────────────────────────────────────────────────────────────

    def add(
        self,
        content: str | Any,
        key: str | None = None,
        *,
        metadata: Dict[str, Any] | None = None,
        embed: bool = True,
    ) -> str:
        """Add content to the store.

        Parameters
        ----------
        content : str | Any
            The content to add. Can be a string or any value that will be serialized.
        key : str | None
            Optional custom key/ID for the item. If None, a key will be auto-generated.
        metadata : Dict[str, Any] | None
            Optional metadata dictionary to associate with the content.
        embed : bool
            Whether to generate embeddings for the content (only if embeddings are enabled).

        Returns
        -------
        str
            The key of the added item.

        Examples
        --------
        ```python
            >>> store = Store()
            >>> store.add("Hello world", key="greeting")
            'greeting'
            >>> store.add("Python programming", metadata={"topic": "tech"})
            'a1b2c3d4e5f6...'
        ```
        """
        return self._run_async(
            self.async_add(
                content, key=key, metadata=metadata, embed=embed
            )
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
    ) -> List[StoreSearchResult]:
        """Search the store for similar content.

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int
            Number of results to return (default: 5).
        filter : Dict[str, Any] | None
            Optional metadata filter (dict of key-value pairs).
        hybrid : bool
            Use hybrid search (vector + keyword). Ignored if embeddings=False.
        hybrid_weight : float
            Weight for vector search (0=keyword only, 1=vector only).

        Returns
        -------
        List[StoreSearchResult]
            List of search results sorted by relevance.

        Examples
        --------
        ```python
            >>> store = Store(embeddings=True)
            >>> store.add("Python is great")
            >>> results = store.search("programming languages", top_k=3)
            >>> print(results[0].item.content)
            'Python is great'
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

    def get(self, key: str) -> StoreItem | None:
        """Get an item by its key.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.

        Returns
        -------
        StoreItem | None
            The store item if found, None otherwise.

        Examples
        --------
        ```python
            >>> store = Store()
            >>> store.add("content", key="mykey")
            >>> item = store.get("mykey")
            >>> print(item.content)
            'content'
        ```
        """
        return self._run_async(self.async_get(key))

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
        ```python
            >>> store = Store()
            >>> store.add("content", key="mykey")
            >>> store.delete("mykey")
            True
            >>> store.get("mykey")
            None
        ```
        """
        return self._run_async(self.async_delete(key))

    def clear(self) -> None:
        """Clear all items from the store.

        Examples
        --------
        ```python
            >>> store = Store()
            >>> store.add("item1")
            >>> store.add("item2")
            >>> store.count()
            2
            >>> store.clear()
            >>> store.count()
            0
        ```
        """
        return self._run_async(self.async_clear())

    def count(self) -> int:
        """Get the total number of unique items in the store.

        Returns
        -------
        int
            Number of unique items (by base key) in the store.

        Examples
        --------
        ```python
            >>> store = Store()
            >>> store.add("item1")
            >>> store.add("item2")
            >>> store.count()
            2
        ```
        """
        return self._run_async(self.async_count())

    # ─────────────────────────────────────────────────────────────────────────────
    # Async API
    # ─────────────────────────────────────────────────────────────────────────────

    def _run_async(self, coro) -> Any:
        """Run an async coroutine in a sync context."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call sync method from a running async loop. "
                    "Use the async variant instead."
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        else:
            return loop.run_until_complete(coro)

    async def async_add(
        self,
        content: str | Any,
        key: str | None = None,
        *,
        metadata: Dict[str, Any] | None = None,
        embed: bool = True,
    ) -> str:
        """Asynchronously add content to the store.

        See `add()` for parameter documentation.
        """
        base_key = self._base_key(key) if key else None

        if key is None:
            key = generate_content_key(content, metadata)
            base_key = key
            _logger.debug("Generated key for content: %s", key)
        elif base_key is None:
            base_key = self._base_key(key)

        text_value, payload = self._serialize_value(content)
        metadata_json = json.dumps(
            metadata or {}, default=str, ensure_ascii=False
        )

        embedding_vector: List[float] | None = None
        if self._settings.embeddings and embed:
            embedding_vector = await self._embed_text(text_value)
            _logger.debug("Generated embedding for key: %s", key)

        if self._settings.deduplicate:
            self._delete_by_base_key(base_key)
            _logger.debug(
                "Deduplicated entries for base_key: %s", base_key
            )

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

    async def async_search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
    ) -> List[StoreSearchResult]:
        """Asynchronously search the store for similar content.

        See `search()` for parameter documentation.
        """
        if self._settings.embeddings and self._settings.embedding_model:
            if hybrid:
                _logger.debug(
                    "Performing hybrid search with weight: %s",
                    hybrid_weight,
                )
                results = await self._hybrid_search(
                    query,
                    top_k=top_k,
                    filter_values=filter,
                    vector_weight=hybrid_weight,
                )
            else:
                _logger.debug("Performing vector search")
                results = await self._vector_search(
                    query, top_k=top_k, filter_values=filter
                )
        else:
            _logger.debug("Performing keyword search")
            results = await self._keyword_search(
                query, top_k=top_k, filter_values=filter
            )

        results = self._deduplicate_results(results)
        results = self._apply_filter(results, filter)
        results.sort(key=lambda r: r.score, reverse=True)
        _logger.debug(
            "Search returned %d results for query: %s",
            len(results[:top_k]),
            query,
        )
        return results[:top_k]

    async def _vector_search(
        self,
        query: str,
        *,
        top_k: int,
        filter_values: Dict[str, Any] | None,
        fetch_multiplier: int = 4,
    ) -> List[StoreSearchResult]:
        """Perform pure vector similarity search."""
        if not self._settings.embedding_model:
            return []

        query_vector = await self._embed_text(query)
        fetch_k = max(top_k * fetch_multiplier, top_k + 8)

        # Optimized: Use SIMD-friendly operations and reduce data transfer
        rows = self._connection.execute(
            f"""
            SELECT id, content, payload, metadata, created_at,
                   list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
            FROM {self._table_name}
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT {fetch_k}
        """,
            [query_vector],
        ).fetchall()

        results: List[StoreSearchResult] = []
        model_name = (
            self._settings.embedding_model.model
            if self._settings.embedding_model
            else None
        )

        # Optimized: Pre-compute values and use local variables
        matches_filter = self._matches_filter
        build_result = self._build_search_result

        for row in rows:
            similarity = float(row[5]) if row[5] is not None else 0.0
            # Optimized: Simplified normalization
            normalized_score = max(0.0, min(1.0, (similarity + 1.0) * 0.5))
            distance = 1.0 - normalized_score

            result = build_result(
                row,
                score=normalized_score,
                distance=distance,
                model_name=model_name,
            )

            if not matches_filter(result.item.metadata, filter_values):
                continue
            results.append(result)

        return results

    async def _keyword_search(
        self,
        query: str,
        *,
        top_k: int,
        filter_values: Dict[str, Any] | None,
        fetch_multiplier: int = 4,
    ) -> List[StoreSearchResult]:
        """Perform pure keyword search using string matching."""
        query_lower = query.lower()
        fetch_k = max(top_k * fetch_multiplier, top_k + 8)

        # Optimized: Compute position only once and use more efficient scoring
        rows = self._connection.execute(
            f"""
            WITH scored AS (
                SELECT
                    id,
                    content,
                    payload,
                    metadata,
                    created_at,
                    POSITION(? IN lower(content)) AS pos
                FROM {self._table_name}
                WHERE POSITION(? IN lower(content)) > 0
            )
            SELECT 
                id, content, payload, metadata, created_at,
                1.0 / pos AS keyword_score
            FROM scored
            ORDER BY keyword_score DESC
            LIMIT {fetch_k}
        """,
            [query_lower, query_lower],
        ).fetchall()

        # Optimized: Use local variables for method calls
        matches_filter = self._matches_filter
        build_result = self._build_search_result

        results: List[StoreSearchResult] = []
        for row in rows:
            keyword_score = row[5]
            score = (
                float(keyword_score) if keyword_score is not None else 0.0
            )
            result = build_result(
                row,
                score=score,
                distance=None,
                model_name=None,
            )
            if not matches_filter(result.item.metadata, filter_values):
                continue
            results.append(result)

        return results

    async def _hybrid_search(
        self,
        query: str,
        *,
        top_k: int,
        filter_values: Dict[str, Any] | None,
        vector_weight: float,
    ) -> List[StoreSearchResult]:
        """Perform hybrid search combining vector and keyword signals."""
        search_limit = max(top_k * 5, top_k + 16)

        # Optimized: Run searches in parallel using asyncio.gather
        keyword_results, vector_results = await asyncio.gather(
            self._keyword_search(
                query, top_k=search_limit, filter_values=filter_values
            ),
            self._vector_search(
                query,
                top_k=search_limit,
                filter_values=filter_values,
                fetch_multiplier=3,
            ),
        )

        # Optimized: Use local variable and pre-compute weights
        base_key = self._base_key
        keyword_weight = 1.0 - vector_weight

        combined: Dict[str, StoreSearchResult] = {}
        vector_map = {
            base_key(result.item.id): result for result in vector_results
        }

        for kw_result in keyword_results:
            bkey = base_key(kw_result.item.id)
            vec_result = vector_map.get(bkey)
            if vec_result:
                blended_score = (
                    keyword_weight * kw_result.score
                    + vector_weight * vec_result.score
                )
                merged = StoreSearchResult(
                    item=kw_result.item,
                    score=blended_score,
                    distance=vec_result.distance,
                    model=vec_result.model,
                )
            else:
                merged = kw_result
            combined[bkey] = merged

        for vec_result in vector_results:
            bkey = base_key(vec_result.item.id)
            if bkey not in combined:
                combined[bkey] = StoreSearchResult(
                    item=vec_result.item,
                    score=vector_weight * vec_result.score,
                    distance=vec_result.distance,
                    model=vec_result.model,
                )

        return list(combined.values())

    async def async_get(self, key: str) -> StoreItem | None:
        """Asynchronously get an item by key.

        See `get()` for parameter documentation.
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

        return StoreItem(
            id=row[0],
            content=row[1],
            metadata=metadata,
            created_at=self._coerce_datetime(row[4]),
            parsed=row[
                2
            ],  # For base Store, parsed is just the raw payload
        )

    async def async_delete(self, key: str) -> bool:
        """Asynchronously delete an item and all its chunks by key.

        See `delete()` for parameter documentation.
        """
        base_key = self._base_key(key)
        rows = self._connection.execute(
            f"DELETE FROM {self._table_name} WHERE base_key = ? RETURNING 1",
            [base_key],
        ).fetchall()
        deleted = len(rows) > 0
        if deleted:
            _logger.debug("Deleted item with key: %s", key)
        return deleted

    async def async_clear(self) -> None:
        """Asynchronously clear all items from the store."""
        self._connection.execute(f"DELETE FROM {self._table_name}")
        _logger.debug("Cleared all items from store")

    async def async_count(self) -> int:
        """Asynchronously get the total number of unique items."""
        row = self._connection.execute(
            f"SELECT COUNT(DISTINCT base_key) FROM {self._table_name}"
        ).fetchone()
        return int(row[0]) if row else 0

    # ─────────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection.

        Examples
        --------
        ```python
            >>> store = Store("data.duckdb")
            >>> store.add("content")
            >>> store.close()
        ```
        """
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "Store":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the store."""
        return (
            f"Store(location={self._settings.location!r}, "
            f"embeddings={self._settings.embeddings}, "
            f"table={self._settings.table_name!r})"
        )
