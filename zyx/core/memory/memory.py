"""zyx.core.memory.memory"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Type, TypeVar

from httpx import URL

from ...core.schemas.schema import Schema
from ...models.embeddings.model import EmbeddingModel, EmbeddingModelName
from ...models.language.model import LanguageModel, LanguageModelName
from ...models.language.model import LanguageModelResponse
from ...utils.processing.text import read_file_as_markdown, read_url_as_markdown
from .types import (
    MemoryDistanceMetric,
    MemoryItem,
    MemoryQueryResponse,
    MemorySearchResult,
    MemorySettings,
)
from .utils import generate_content_key, json_ready, stable_string

__all__ = ["Memory", "mem", "rag", "arag"]

_logger = logging.getLogger(__name__)

DEFAULT_TABLE_NAME = "memory_items"

T = TypeVar("T")


@dataclass
class _TypeEntry:
    """Internal representation of a type entry in the type registry."""

    factory: Any
    schema: Schema[Any] | None


def _load_duckdb():
    """Lazily load DuckDB with helpful error message."""
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "Could not import duckdb package. "
            "Please install it with `pip install duckdb`."
        ) from exc
    return duckdb


class Memory:
    """DuckDB-based memory system with optional embeddings and type-safe storage.

    Features:
    - Synchronous methods (e.g., `memory.add()`) for standard Python.
    - Asynchronous methods (e.g., `await memory.async_add()`) for async apps.
    - Optional embedding generation for semantic search.
    - Keyword search using DuckDB's string functions.
    - Hybrid search combining vector and keyword signals.
    - Type-safe storage with automatic serialization/deserialization.
    - Automatic deduplication by content key.

    Example (Sync):
        >>> memory = Memory(embeddings=True)
        >>> memory.add("Steve loves Python")
        >>> results = memory.search("What does Steve like?")

    Example (Async):
        >>> memory = Memory(embeddings=True)
        >>> await memory.async_add("Steve loves Python")
        >>> results = await memory.async_search("What does Steve like?")

    Example (Typed):
        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class Person:
        ...     name: str
        ...     age: int
        >>> memory = Memory(types={"person": Person})
        >>> memory.add(Person("Alice", 30), key="person")
        >>> item = memory.get("person")
        >>> print(item.parsed)  # Person(name='Alice', age=30)
    """

    _connection: Any | None = None

    @property
    def settings(self) -> MemorySettings:
        """Memory configuration settings."""
        return self._settings

    def __init__(
        self,
        location: Path | str | Literal[":memory:"] = ":memory:",
        *,
        embeddings: bool = True,
        embedding_model: EmbeddingModelName | str | EmbeddingModel | None = None,
        distance_metric: MemoryDistanceMetric = "cosine",
        deduplicate: bool = True,
        types: Dict[str, Any] | Any = str,
        connection: Any | None = None,
        table_name: str = DEFAULT_TABLE_NAME,
    ):
        """Initialize the Memory interface.

        Parameters
        ----------
        location : Path | str | Literal[":memory:"]
            Path or identifier for the memory database. Use ":memory:" for
            in-memory database.
        embeddings : bool
            Whether to enable embedding generation and search.
        embedding_model : EmbeddingModelName | str | EmbeddingModel | None
            Embedding model to use (if embeddings=True). Can be a model name
            or an EmbeddingModel instance.
        distance_metric : MemoryDistanceMetric
            Distance metric for similarity (if embeddings=True).
        deduplicate : bool
            Whether to deduplicate items by key on add/search.
        types : Dict[str, Any] | Any
            Type specification for keys. Can be:
            - A single type (e.g., `str`, `int`, `MyModel`) for all keys
            - A dict mapping keys to types (e.g., `{"person": Person, "age": int}`)
            - A dict with `"__default__"` key for fallback type
        connection : Any | None
            Optional existing DuckDB connection to use.
        table_name : str
            Name of the table to use for storing memory items.
        """
        self._duckdb = _load_duckdb()

        if embeddings:
            if embedding_model is None:
                embedding_model = "openai/text-embedding-3-small"
                _logger.debug(
                    "Embeddings enabled but no model provided, defaulting to %s",
                    embedding_model,
                )

            if isinstance(embedding_model, EmbeddingModel):
                resolved_embedding_model = embedding_model
            else:
                resolved_embedding_model = EmbeddingModel(embedding_model)
        else:
            resolved_embedding_model = None

        self._type_map, self._default_type_entry = self._normalize_type_spec(types)

        location_str = str(location)
        self._settings = MemorySettings(
            location=location_str,
            embeddings=embeddings,
            embedding_model=resolved_embedding_model,
            distance_metric=distance_metric,
            deduplicate=deduplicate,
            types=types,
            table_name=table_name,
        )

        self._connection = connection or self._duckdb.connect(
            database=location_str,
            config={
                "enable_external_access": "false",
                "autoload_known_extensions": "false",
                "autoinstall_known_extensions": "false",
            },
        )
        self._table_name = table_name
        self._ensure_table()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_type_spec(
        self, spec: Dict[str, Any] | Any
    ) -> tuple[Dict[str, _TypeEntry], _TypeEntry]:
        """Normalize type specification into internal format."""
        if isinstance(spec, dict):
            default = spec.get("__default__", str)
            mapping = {
                key: self._build_type_entry(value)
                for key, value in spec.items()
                if key != "__default__"
            }
            return mapping, self._build_type_entry(default)
        return {}, self._build_type_entry(spec)

    def _build_type_entry(self, value: Any) -> _TypeEntry:
        """Build a type entry from a value."""
        if isinstance(value, Schema):
            return _TypeEntry(factory=value, schema=value)
        if value is str:
            return _TypeEntry(factory=str, schema=None)
        try:
            schema = Schema(value)
        except Exception:
            schema = None
        return _TypeEntry(factory=value, schema=schema)

    def _ensure_table(self) -> None:
        """Ensure the memory table exists."""
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

    def _base_key(self, key: str) -> str:
        """Extract base key from a full key (handles chunked keys)."""
        if "::" in key:
            return key.split("::", 1)[0]
        if ":" in key:
            return key.split(":", 1)[0]
        return key

    def _delete_by_base_key(self, base_key: str) -> None:
        """Delete all entries with the given base key."""
        self._connection.execute(
            f"DELETE FROM {self._table_name} WHERE base_key = ?", [base_key]
        )

    def _serialize_value(self, base_key: str, value: Any) -> tuple[str, str, Any]:
        """Serialize a value according to its type entry.

        Returns
        -------
        tuple[str, str, Any]
            (text_value, payload, json_ready_value)
        """
        entry = self._type_map.get(base_key, self._default_type_entry)
        if entry.schema:
            try:
                value = entry.schema(value)
            except Exception as exc:
                raise ValueError(
                    f"Value does not conform to schema for key '{base_key}'."
                ) from exc

        json_ready_value = json_ready(value)
        payload = self._serialize_payload(json_ready_value)
        text_value = stable_string(json_ready_value)
        return text_value, payload, json_ready_value

    def _serialize_payload(self, value: Any) -> str:
        """Serialize a value to JSON string."""
        try:
            return json.dumps(value, default=str, ensure_ascii=False)
        except TypeError:  # pragma: no cover
            return json.dumps(str(value), ensure_ascii=False)

    def _deserialize_payload(self, base_key: str, payload: str | None) -> Any:
        """Deserialize a payload according to its type entry."""
        if payload is None:
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = payload

        entry = self._type_map.get(base_key, self._default_type_entry)
        if entry.schema:
            try:
                return entry.schema(data)
            except Exception:
                pass

        factory = entry.factory
        if factory in (None, str):
            return str(data) if not isinstance(data, str) else data
        try:
            return factory(data)
        except Exception:
            return data

    def _metadata_from_json(self, metadata_json: str | None) -> Dict[str, Any]:
        """Parse metadata from JSON string."""
        if not metadata_json:
            return {}
        try:
            value = json.loads(metadata_json)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _matches_filter(
        self, metadata: Dict[str, Any], filter_values: Dict[str, Any] | None
    ) -> bool:
        """Check if metadata matches filter criteria."""
        if not filter_values:
            return True
        return all(metadata.get(key) == value for key, value in filter_values.items())

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
        self, results: List[MemorySearchResult]
    ) -> List[MemorySearchResult]:
        """Deduplicate search results by base key, keeping highest score."""
        if not self._settings.deduplicate:
            return results

        deduped: Dict[str, MemorySearchResult] = {}
        for result in results:
            base_key = self._base_key(result.item.id)
            existing = deduped.get(base_key)
            if existing is None or result.score > existing.score:
                deduped[base_key] = result
        return list(deduped.values())

    def _build_search_result(
        self,
        row: Any,
        score: float,
        distance: float | None,
        model_name: str | None,
    ) -> MemorySearchResult:
        """Build a MemorySearchResult from a database row."""
        (
            item_id,
            content,
            payload,
            metadata_json,
            created_at_raw,
        ) = row[:5]

        metadata = self._metadata_from_json(metadata_json)
        parsed_value = self._deserialize_payload(self._base_key(item_id), payload)

        item = MemoryItem(
            id=item_id,
            content=content,
            metadata=metadata,
            created_at=self._coerce_datetime(created_at_raw),
            parsed=parsed_value,
        )

        return MemorySearchResult(
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

    async def _resolve_content(self, content: Any) -> Any:
        """Resolve content from Path, URL, or string to actual content."""
        if isinstance(content, Path):
            return read_file_as_markdown(str(content))
        if isinstance(content, URL):
            return read_url_as_markdown(str(content))
        if isinstance(content, str):
            if content.startswith("http://") or content.startswith("https://"):
                try:
                    return read_url_as_markdown(content)
                except Exception:
                    return content
            potential_path = Path(content)
            if potential_path.is_file():
                try:
                    return read_file_as_markdown(content)
                except Exception:
                    return content
        return content

    def _apply_filter(
        self, results: List[MemorySearchResult], filter_values: Dict[str, Any] | None
    ) -> List[MemorySearchResult]:
        """Apply metadata filter to search results."""
        if not filter_values:
            return results
        return [
            result
            for result in results
            if self._matches_filter(result.item.metadata, filter_values)
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API (sync wrappers)
    # ─────────────────────────────────────────────────────────────────────────

    def add(
        self,
        content: str | list[str] | Path | URL | Any,
        key: str | None = None,
        *,
        metadata: Dict[str, Any] | None = None,
        embed: bool = True,
    ) -> str | list[str]:
        """Synchronously add content to memory.

        Parameters
        ----------
        content : str | list[str] | Path | URL | Any
            The content to add. Can be a string, list of strings, file Path,
            URL, or any typed value matching the type specification.
        key : str | None
            Optional custom key/ID for the memory item. If None, a key will
            be auto-generated from the content.
        metadata : Dict[str, Any] | None
            Optional metadata dictionary to associate with the content.
        embed : bool
            Whether to generate embeddings for the content. Only respected if
            `self.settings.embeddings` is True.

        Returns
        -------
        str | list[str]
            The key or list of keys of the added memory item(s).
        """
        return self._run_async(
            self.async_add(content, key=key, metadata=metadata, embed=embed)
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
    ) -> list[MemorySearchResult]:
        """Synchronously search memory for similar content.

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int
            Number of results to return.
        filter : Dict[str, Any] | None
            Optional metadata filter (dict of key-value pairs).
        hybrid : bool
            Use hybrid search (vector + keyword). Ignored if embeddings=False.
        hybrid_weight : float
            Weight for vector search (0=keyword only, 1=vector only).

        Returns
        -------
        list[MemorySearchResult]
            List of search results sorted by relevance.
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

    def get(self, key: str) -> MemoryItem | None:
        """Synchronously get item by key.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.

        Returns
        -------
        MemoryItem | None
            The memory item if found, None otherwise.
        """
        return self._run_async(self.async_get(key))

    def delete(self, key: str) -> bool:
        """Synchronously delete item and all its chunks by key.

        Parameters
        ----------
        key : str
            The key of the item to delete.

        Returns
        -------
        bool
            True if the item was deleted, False otherwise.
        """
        return self._run_async(self.async_delete(key))

    def clear(self) -> None:
        """Synchronously clear all memories."""
        return self._run_async(self.async_clear())

    def count(self) -> int:
        """Get total number of unique memory items.

        Returns
        -------
        int
            Number of unique items (by base key) in memory.
        """
        return self._run_async(self.async_count())

    def rag(
        self,
        query: str,
        language_model: LanguageModelName | str | LanguageModel = "openai/gpt-4o-mini",
        *,
        type: Type[T] | Schema[T] = str,
        prompt_template: str | None = None,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
        **kwargs,
    ) -> MemoryQueryResponse[T]:
        """Synchronously search memory and query a language model with results.

        Parameters
        ----------
        query : str
            The user's query to search for and answer.
        language_model : LanguageModelName | str | LanguageModel
            Language model to use (name or LanguageModel instance).
        type : Type[T] | Schema[T]
            The expected response type (Pydantic model, Schema, or `str`).
        prompt_template : str | None
            Optional f-string template for the prompt. Must include `{context}`
            and `{query}` placeholders.
        top_k : int
            Number of search results to retrieve.
        filter : Dict[str, Any] | None
            Metadata filter for the search.
        hybrid : bool
            Whether to use hybrid search (if embeddings are enabled).
        hybrid_weight : float
            The weight for vector search in hybrid mode.
        **kwargs
            Additional keyword arguments to pass to `language_model.run()`.

        Returns
        -------
        MemoryQueryResponse[T]
            A response object containing the LLM response and search results.
        """
        return self._run_async(
            self.async_rag(
                query,
                language_model,
                type=type,
                prompt_template=prompt_template,
                top_k=top_k,
                filter=filter,
                hybrid=hybrid,
                hybrid_weight=hybrid_weight,
                **kwargs,
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Async API
    # ─────────────────────────────────────────────────────────────────────────

    def _run_async(self, coro: asyncio.Future) -> Any:
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
        content: str | list[str] | Path | URL | Any,
        key: str | None = None,
        *,
        metadata: Dict[str, Any] | None = None,
        embed: bool = True,
    ) -> str | list[str]:
        """Asynchronously add content to memory.

        See `add()` for parameter documentation.
        """
        if isinstance(content, list):
            results = []
            for item in content:
                result = await self.async_add(
                    item, key=None, metadata=metadata, embed=embed
                )
                results.append(result)
            return results

        processed_content = await self._resolve_content(content)
        base_key = self._base_key(key) if key else None

        if key is None:
            key = generate_content_key(processed_content, metadata)
            base_key = key
        elif base_key is None:
            base_key = self._base_key(key)

        text_value, payload, _ = self._serialize_value(base_key, processed_content)
        metadata_json = json.dumps(metadata or {}, default=str, ensure_ascii=False)

        embedding_vector: List[float] | None = None
        if self._settings.embeddings and embed:
            embedding_vector = await self._embed_text(text_value)

        if self._settings.deduplicate:
            self._delete_by_base_key(base_key)

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
    ) -> List[MemorySearchResult]:
        """Asynchronously search memory for similar content.

        See `search()` for parameter documentation.
        """
        if self._settings.embeddings and self._settings.embedding_model:
            if hybrid:
                results = await self._hybrid_search(
                    query,
                    top_k=top_k,
                    filter_values=filter,
                    vector_weight=hybrid_weight,
                )
            else:
                results = await self._vector_search(
                    query, top_k=top_k, filter_values=filter
                )
        else:
            results = await self._keyword_search(
                query, top_k=top_k, filter_values=filter
            )

        results = self._deduplicate_results(results)
        results = self._apply_filter(results, filter)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def _vector_search(
        self,
        query: str,
        *,
        top_k: int,
        filter_values: Dict[str, Any] | None,
        fetch_multiplier: int = 4,
    ) -> List[MemorySearchResult]:
        """Perform pure vector similarity search."""
        if not self._settings.embedding_model:
            return []

        query_vector = await self._embed_text(query)
        fetch_k = max(top_k * fetch_multiplier, top_k + 8)

        rows = self._connection.execute(
            f"""
            SELECT id, content, payload, metadata, created_at,
                   list_cosine_similarity(embedding, ?) AS similarity
            FROM {self._table_name}
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT {fetch_k}
        """,
            [query_vector],
        ).fetchall()

        results: List[MemorySearchResult] = []
        model_name = (
            self._settings.embedding_model.model
            if self._settings.embedding_model
            else None
        )

        for row in rows:
            similarity = float(row[5]) if row[5] is not None else 0.0
            normalized_score = max(min((similarity + 1.0) / 2.0, 1.0), 0.0)
            distance = 1.0 - normalized_score

            result = self._build_search_result(
                row,
                score=normalized_score,
                distance=distance,
                model_name=model_name,
            )

            if not self._matches_filter(result.item.metadata, filter_values):
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
    ) -> List[MemorySearchResult]:
        """Perform pure keyword search using string matching."""
        query_lower = query.lower()
        fetch_k = max(top_k * fetch_multiplier, top_k + 8)

        rows = self._connection.execute(
            f"""
            SELECT
                id,
                content,
                payload,
                metadata,
                created_at,
                1.0 / POSITION(? IN lower(content)) AS keyword_score
            FROM {self._table_name}
            WHERE POSITION(? IN lower(content)) > 0
            ORDER BY keyword_score DESC
            LIMIT {fetch_k}
        """,
            [query_lower, query_lower],
        ).fetchall()

        results: List[MemorySearchResult] = []
        for row in rows:
            keyword_score = row[5]
            score = float(keyword_score) if keyword_score is not None else 0.0
            result = self._build_search_result(
                row,
                score=score,
                distance=None,
                model_name=None,
            )
            if not self._matches_filter(result.item.metadata, filter_values):
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
    ) -> List[MemorySearchResult]:
        """Perform hybrid search combining vector and keyword signals."""
        search_limit = max(top_k * 5, top_k + 16)
        keyword_results = await self._keyword_search(
            query, top_k=search_limit, filter_values=filter_values
        )
        vector_results = await self._vector_search(
            query,
            top_k=search_limit,
            filter_values=filter_values,
            fetch_multiplier=3,
        )

        combined: Dict[str, MemorySearchResult] = {}
        vector_map = {
            self._base_key(result.item.id): result for result in vector_results
        }

        for kw_result in keyword_results:
            base_key = self._base_key(kw_result.item.id)
            vec_result = vector_map.get(base_key)
            if vec_result:
                blended_score = (
                    1.0 - vector_weight
                ) * kw_result.score + vector_weight * vec_result.score
                merged = MemorySearchResult(
                    item=kw_result.item,
                    score=blended_score,
                    distance=vec_result.distance,
                    model=vec_result.model,
                )
            else:
                merged = kw_result
            combined[base_key] = merged

        for vec_result in vector_results:
            base_key = self._base_key(vec_result.item.id)
            if base_key not in combined:
                combined[base_key] = MemorySearchResult(
                    item=vec_result.item,
                    score=vector_weight * vec_result.score,
                    distance=vec_result.distance,
                    model=vec_result.model,
                )

        return list(combined.values())

    async def async_get(self, key: str) -> MemoryItem | None:
        """Asynchronously get item by key.

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
        parsed = self._deserialize_payload(self._base_key(row[0]), row[2])

        return MemoryItem(
            id=row[0],
            content=row[1],
            metadata=metadata,
            created_at=self._coerce_datetime(row[4]),
            parsed=parsed,
        )

    async def async_delete(self, key: str) -> bool:
        """Asynchronously delete item and all its chunks by key.

        See `delete()` for parameter documentation.
        """
        base_key = self._base_key(key)
        rows = self._connection.execute(
            f"DELETE FROM {self._table_name} WHERE base_key = ? RETURNING 1",
            [base_key],
        ).fetchall()
        return len(rows) > 0

    async def async_clear(self) -> None:
        """Asynchronously clear all memories."""
        self._connection.execute(f"DELETE FROM {self._table_name}")

    async def async_count(self) -> int:
        """Asynchronously get total number of unique memory items."""
        row = self._connection.execute(
            f"SELECT COUNT(DISTINCT base_key) FROM {self._table_name}"
        ).fetchone()
        return int(row[0]) if row else 0

    async def async_rag(
        self,
        query: str,
        language_model: LanguageModelName | str | LanguageModel,
        *,
        type: Type[T] | Schema[T] = str,
        prompt_template: str | None = None,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
        **kwargs,
    ) -> MemoryQueryResponse[T]:
        """Asynchronously search memory and query a language model with results.

        See `rag()` for parameter documentation.
        """
        if not isinstance(language_model, LanguageModel):
            language_model = LanguageModel(language_model)

        search_results = await self.async_search(
            query,
            top_k=top_k,
            filter=filter,
            hybrid=hybrid,
            hybrid_weight=hybrid_weight,
        )

        if not search_results:
            context_str = "No relevant context was found in memory."
        else:
            context_str = "\n\n".join(
                f"Context Item (ID: {result.item.id}):\n{result.item.content}"
                for result in search_results
            )

        prompt_template = prompt_template or (
            "Use the following context to answer the user's query.\n\n"
            "Context:\n{context}\n\nUser Query: {query}\nAnswer:"
        )

        try:
            prompt = prompt_template.format(context=context_str, query=query)
        except KeyError as exc:
            raise ValueError(
                "Invalid prompt template. Ensure it contains {context} and {query}."
            ) from exc

        if "stream" in kwargs and kwargs["stream"]:
            kwargs["stream"] = False

        response: LanguageModelResponse[T] = await language_model.arun(
            prompt, type=type, **kwargs
        )

        return MemoryQueryResponse(response=response, results=search_results)

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __str__(self) -> str:
        from ..._lib._beautification import _pretty_print_memory

        return _pretty_print_memory(self)

    def __rich__(self):
        from ..._lib._beautification import _rich_pretty_print_memory

        return _rich_pretty_print_memory(self)


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────


def mem(
    location: Path | str | Literal[":memory:"] = ":memory:",
    *,
    embeddings: bool = True,
    embedding_model: EmbeddingModelName | str | EmbeddingModel | None = None,
    distance_metric: MemoryDistanceMetric = "cosine",
    deduplicate: bool = True,
    types: Dict[str, Any] | Any = str,
    connection: Any | None = None,
    table_name: str = DEFAULT_TABLE_NAME,
) -> Memory:
    """Factory function to create a Memory instance.

    Parameters
    ----------
    See `Memory.__init__()` for parameter documentation.

    Returns
    -------
    Memory
        An initialized Memory instance.
    """
    return Memory(
        location=location,
        embeddings=embeddings,
        embedding_model=embedding_model,
        distance_metric=distance_metric,
        deduplicate=deduplicate,
        types=types,
        connection=connection,
        table_name=table_name,
    )


def _get_rag_db_hash(items: Any | List[Any]) -> str:
    """Generate a stable hash for RAG cache key."""
    if not isinstance(items, list):
        items = [items]
    item_tuple = tuple(sorted(str(item) for item in items))
    digest = hashlib.sha256(str(item_tuple).encode("utf-8")).hexdigest()
    return digest[:16]


@lru_cache(maxsize=16)
def _get_cached_rag_memory_instance(
    location: str,
    embeddings: bool,
    embedding_model_name: str | None,
    distance_metric: MemoryDistanceMetric,
    deduplicate: bool,
    types_hash: str,
) -> Memory:
    """Internal LRU-cached function to create/get Memory instances."""
    return Memory(
        location=location,
        embeddings=embeddings,
        embedding_model=embedding_model_name,
        distance_metric=distance_metric,
        deduplicate=deduplicate,
        types=str,  # Default to str for cached RAG instances
    )


async def arag(
    items: Any | List[Any],
    query: str,
    *,
    language_model: LanguageModelName | str | LanguageModel = "openai/gpt-4o-mini",
    type: Type[T] | Schema[T] = str,
    embeddings: bool = True,
    embedding_model: EmbeddingModelName | str | EmbeddingModel | None = None,
    distance_metric: MemoryDistanceMetric = "cosine",
    deduplicate: bool = True,
    types: Dict[str, Any] | Any = str,
    top_k: int = 5,
    filter: Dict[str, Any] | None = None,
    hybrid: bool = True,
    hybrid_weight: float = 0.5,
    **kwargs,
) -> MemoryQueryResponse[T]:
    """Asynchronously perform RAG over a list of items.

    Parameters
    ----------
    items : Any | List[Any]
        A single content string/Path/URL or a list of them to add to memory.
    query : str
        The user's query to search for and answer.
    language_model : LanguageModelName | str | LanguageModel
        Language model to use (name or LanguageModel instance).
    type : Type[T] | Schema[T]
        The expected response type (Pydantic model, Schema, or `str`).
    embeddings : bool
        Whether to enable embedding generation and search.
    embedding_model : EmbeddingModelName | str | EmbeddingModel | None
        Embedding model to use (if embeddings=True).
    distance_metric : MemoryDistanceMetric
        Distance metric for similarity (if embeddings=True).
    deduplicate : bool
        Whether to deduplicate items.
    types : Dict[str, Any] | Any
        Type specification for keys.
    top_k : int
        Number of search results to retrieve.
    filter : Dict[str, Any] | None
        Metadata filter for the search.
    hybrid : bool
        Whether to use hybrid search (if embeddings are enabled).
    hybrid_weight : float
        The weight for vector search in hybrid mode.
    **kwargs
        Additional keyword arguments to pass to `language_model.arun()`.

    Returns
    -------
    MemoryQueryResponse[T]
        A response object containing the LLM response and search results.
    """
    db_hash = _get_rag_db_hash(items)
    location = f"zyx_rag_cache_{db_hash}.duckdb"

    embedding_model_name = None
    if isinstance(embedding_model, EmbeddingModel):
        embedding_model_name = embedding_model.model
    elif isinstance(embedding_model, str):
        embedding_model_name = embedding_model

    types_hash = hashlib.sha256(str(types).encode()).hexdigest()[:16]

    memory_instance = _get_cached_rag_memory_instance(
        location=location,
        embeddings=embeddings,
        embedding_model_name=embedding_model_name,
        distance_metric=distance_metric,
        deduplicate=deduplicate,
        types_hash=types_hash,
    )

    if await memory_instance.async_count() == 0:
        items_list = items if isinstance(items, list) else [items]
        for item in items_list:
            await memory_instance.async_add(item, embed=embeddings)

    return await memory_instance.async_rag(
        query,
        language_model,
        type=type,
        top_k=top_k,
        filter=filter,
        hybrid=hybrid,
        hybrid_weight=hybrid_weight,
        **kwargs,
    )


def rag(
    items: Any | List[Any],
    query: str,
    *,
    language_model: LanguageModelName | str | LanguageModel = "openai/gpt-4o-mini",
    type: Type[T] | Schema[T] = str,
    embeddings: bool = True,
    embedding_model: EmbeddingModelName | str | EmbeddingModel | None = None,
    distance_metric: MemoryDistanceMetric = "cosine",
    deduplicate: bool = True,
    types: Dict[str, Any] | Any = str,
    top_k: int = 5,
    filter: Dict[str, Any] | None = None,
    hybrid: bool = True,
    hybrid_weight: float = 0.5,
    **kwargs,
) -> MemoryQueryResponse[T]:
    """Synchronously perform RAG over a list of items.

    See `arag()` for parameter documentation.
    """
    return asyncio.run(
        arag(
            items,
            query,
            language_model=language_model,
            type=type,
            embeddings=embeddings,
            embedding_model=embedding_model,
            distance_metric=distance_metric,
            deduplicate=deduplicate,
            types=types,
            top_k=top_k,
            filter=filter,
            hybrid=hybrid,
            hybrid_weight=hybrid_weight,
            **kwargs,
        )
    )
