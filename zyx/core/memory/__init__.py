"""zyx.core.memory"""

from __future__ import annotations

import sqlite3
import asyncio
from functools import lru_cache
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Any, Dict, Protocol, Type, TypeVar, Coroutine, TYPE_CHECKING

import sqlite_vec

from .._logging import _get_logger
from ...models.embeddings.types import (
    EmbeddingModelName,
)
from ...models.embeddings.model import EmbeddingModel
from ...models.language.model import LanguageModel
from ...models.language.types import LanguageModelResponse, LanguageModelName
from .types import (
    MemoryDistanceMetric,
    MemorySettings,
    MemoryItem,
    MemorySearchResult,
    MemoryQueryResponse,
)
from .utils import (
    serialize_float32_vector_to_bytes,
    deserialize_bytes_to_float32_vector,
    generate_content_id,
)

if TYPE_CHECKING:
    from chonkie.chunker.base import BaseChunker
    from chonkie.chunker.recursive import RecursiveChunker


T_ResponseModel = TypeVar("T_ResponseModel")
"""Generic TypeVar for the MemoryQueryResponse model."""

DEFAULT_QUERY_PROMPT = (
    "Use the following pieces of context to answer the user's query.\n"
    "If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n\n"
    "User Query: {query}\n"
    "Answer:"
)


@lru_cache(maxsize=1)
def cached_recusrive_chunker() -> RecursiveChunker:
    """Lazily load and cache the default RecursiveChunker."""
    from chonkie.chunker.recursive import RecursiveChunker

    return RecursiveChunker(chunk_size=2048)


class Memory:
    """
    SQLite-vec based memory system with a dual sync/async API.

    Features:
    - Synchronous methods (e.g., `memory.add()`) for standard Python.
    - Asynchronous methods (e.g., `await memory.async_add()`) for async apps.
    - Text-only (keyword) or Embedding (semantic) modes.
    - Automatic chunking of long content.
    - Hybrid, vector, and keyword search.
    - `query()` method to search and feed results to a LanguageModel.

    Example (Sync):
        memory = Memory(embeddings=True)
        memory.add("Steve loves Python")
        results = memory.search("What does Steve like?")

    Example (Async):
        memory = Memory(embeddings=True)
        await memory.async_add("Steve loves Python")
        results = await memory.async_search("What does Steve like?")
    """

    _connection: sqlite3.Connection | None = None

    @property
    def settings(self) -> MemorySettings:
        """Memory configuration settings."""
        return self._settings

    def __init__(
        self,
        location: Path | str | Literal[":memory:"] = ":memory:",
        *,
        embeddings: bool = False,
        embedding_model: EmbeddingModelName | str | EmbeddingModel | None = None,
        dimensions: int = 1536,
        chunk_size: int = 2048,
        distance_metric: MemoryDistanceMetric = "cosine",
        chunker: BaseChunker | None = None,
    ):
        """Initialize the Memory interface.

        Args:
            location: Path or identifier for the memory database.
                Use ":memory:" for in-memory database.
            embeddings: Whether to enable embedding generation and search.
                If False, memory is text-only (keyword search).
            embedding_model: Embedding model to use (if embeddings=True).
                Can be a model name or an EmbeddingModel instance.
            dimensions: Number of dimensions (if embeddings=True).
            chunk_size: Maximum chunk size for content.
            distance_metric: Distance metric for similarity (if embeddings=True).
            chunker: Optional custom chunker for splitting long content.
        """
        self.logger = _get_logger(self.__class__.__name__)

        if sqlite_vec is None:
            raise ImportError(
                "sqlite_vec is required. Install with: pip install sqlite-vec"
            )

        # 1. Handle conditional embedding model initialization
        initialized_embedding_model: EmbeddingModel | None = None
        if embeddings:
            if embedding_model is None:
                # Default to a standard model if embeddings are on but none provided
                embedding_model = "openai/text-embedding-3-small"
                self.logger.debug(
                    "Embeddings enabled but no model provided, defaulting to %s",
                    embedding_model,
                )

            if not isinstance(embedding_model, EmbeddingModel):
                try:
                    initialized_embedding_model = EmbeddingModel(
                        embedding_model, adapter="auto"
                    )
                except Exception as e:
                    self.logger.error(
                        "Failed to initialize EmbeddingModel for memory: %s", str(e)
                    )
                    raise
            else:
                initialized_embedding_model = embedding_model

        # 2. Store Settings
        self._settings = MemorySettings(
            location=location,
            embeddings=embeddings,
            embedding_model=initialized_embedding_model,
            dimensions=dimensions,
            chunk_size=chunk_size,
            distance_metric=distance_metric,
            chunker=chunker if chunker is not None else cached_recusrive_chunker(),
        )

        # 3. Initialize Database and Tables
        self._initialize_database()

    def _initialize_database(self):
        """Initializes the SQLite database connection and tables."""
        self._connection = sqlite3.connect(
            self.settings.location, check_same_thread=False
        )
        self._connection.enable_load_extension(True)

        # Load sqlite-vec extension (only needed if embeddings are on)
        if self.settings.embeddings:
            sqlite_vec.load(self._connection)
            self.logger.debug(
                "Loaded sqlite-vec extension for memory database at %s",
                self.settings.location,
            )

        self._connection.enable_load_extension(False)

        # Create main content table (always)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                chunk_index INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create FTS5 table for keyword search (always)
        self._connection.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                item_id,
                content,
                tokenize='porter unicode61'
            )
        """)

        # Create vector table (conditionally)
        if self.settings.embeddings:
            self._connection.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                    embedding float[{self.settings.dimensions}],
                    +item_id TEXT,
                    +chunk_index INTEGER
                )
            """)

        # Create indices
        self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_items_id
            ON memory_items(id)
        """)

        self._connection.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying SQLite connection."""
        if self._connection is None:
            # Re-initialize if connection was closed
            self._initialize_database()
        return self._connection

    def _generate_id(self, content: str, metadata: dict | None = None) -> str:
        """Helper to generate a unique content ID."""
        return generate_content_id(content, metadata)

    def _get_embedding_dimensions(self) -> int | None:
        """Get the dimensions of existing embeddings in the database."""
        cursor = self.connection.execute("SELECT COUNT(*) FROM memory_vectors")
        if cursor.fetchone()[0] == 0:
            return None
        # Get one embedding and check its length
        cursor = self.connection.execute("SELECT embedding FROM memory_vectors LIMIT 1")
        row = cursor.fetchone()
        if row:
            embedding_list = deserialize_bytes_to_float32_vector(row[0])
            return len(embedding_list)
        return None

    def _run_async(self, coro: Coroutine) -> Any:
        """
        Runs an async coroutine in a sync context, handling the event loop.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # This is a critical error. We can't block an already-running loop.
                self.logger.error(
                    "Cannot call sync method from a running async loop. Use the 'async_*' method instead."
                )
                raise RuntimeError(
                    "Cannot use sync method when an event loop is already running. "
                    "Use the async variant (e.g., `await memory.async_add()`) instead."
                )
        except RuntimeError:
            # No event loop is running. Create a new one.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    # --- Synchronous API Methods ---

    def add(
        self,
        content: str | list[str],
        metadata: Dict[str, Any] | None = None,
        chunk: bool = True,
        embed: bool = True,
        item_id: str | None = None,
    ) -> str | list[str]:
        """
        Synchronously add content to the memory.

        Args:
            content: The content string or list of strings to add.
            metadata: Optional metadata dictionary to associate with the content.
            chunk: Whether to chunk the content if it exceeds chunk_size.
            embed: Whether to generate embeddings for the content.
                   This is only respected if `self.settings.embeddings` is True.
            item_id: Optional custom ID for the memory item(s).
                If None, IDs will be auto-generated.

        Returns:
            The ID or list of IDs of the added memory item(s).
        """
        return self._run_async(self.async_add(content, metadata, chunk, embed, item_id))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
    ) -> list[MemorySearchResult]:
        """
        Synchronously search memory for similar content.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            filter: Optional metadata filter (dict of key-value pairs).
            hybrid: Use hybrid search (vector + keyword). Ignored if embeddings=False.
            hybrid_weight: Weight for vector search (0=keyword, 1=vector).

        Returns:
            List of SearchResult objects.
        """
        return self._run_async(
            self.async_search(query, top_k, filter, hybrid, hybrid_weight)
        )

    def get(self, item_id: str) -> MemoryItem | None:
        """Synchronously get item by ID."""
        return self._run_async(self.async_get(item_id))

    def delete(self, item_id: str) -> bool:
        """Synchronously delete item and all its chunks by ID."""
        return self._run_async(self.async_delete(item_id))

    def clear(self):
        """Synchronously clear all memories."""
        return self._run_async(self.async_clear())

    def rag(
        self,
        query: str,
        language_model: LanguageModelName | str | LanguageModel = "openai/gpt-4o-mini",
        *,
        type: Type[T_ResponseModel] = str,
        prompt_template: str | None = None,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
        **kwargs,
    ) -> MemoryQueryResponse[T_ResponseModel]:
        """
        Synchronously search memory and then query a language model with the results.

        Args:
            query: The user's query to search for and answer.
            language_model: An initialized `zyx.models.LanguageModel` instance.
            type: The expected Pydantic model for a structured response,
                  or `str` (default) for a text response.
            prompt_template: An optional f-string template for the prompt.
                             Must include `{context}` and `{query}` placeholders.
            top_k: Number of search results to retrieve.
            filter: Metadata filter for the search.
            hybrid: Whether to use hybrid search (if embeddings are enabled).
            hybrid_weight: The weight for vector search in hybrid mode.
            **kwargs: Additional keyword arguments to pass to `language_model.run()`.

        Returns:
            A `MemoryQueryResponse` object.
        """
        # Note: The async_rag method uses language_model.arun().
        # We wrap the *entire* async_rag operation.
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

    # --- Asynchronous API Methods ---

    async def async_add(
        self,
        content: str | list[str],
        metadata: Dict[str, Any] | None = None,
        chunk: bool = True,
        embed: bool = True,
        item_id: str | None = None,
    ) -> str | list[str]:
        """
        Asynchronously add content to the memory.

        Args:
            content: The content string or list of strings to add.
            metadata: Optional metadata dictionary to associate with the content.
            chunk: Whether to chunk the content if it exceeds chunk_size.
            embed: Whether to generate embeddings for the content.
                   This is only respected if `self.settings.embeddings` is True.
            item_id: Optional custom ID for the memory item(s).
                If None, IDs will be auto-generated.

        Returns:
            The ID or list of IDs of the added memory item(s).
        """
        if isinstance(content, list):
            ids = []
            for text in content:
                # Note: chunk, embed and metadata are passed for each item in the list
                id = await self.async_add(text, metadata, chunk, embed, None)
                ids.append(id)
            return ids

        # Generate ID
        if item_id is None:
            item_id = self._generate_id(content, metadata)

        # Chunk if needed
        chunks = []
        if chunk:
            try:
                # The RecursiveChunker from chonkie returns objects with a 'text' attribute
                chunk_objects = self.settings.chunker.chunk(content)
                chunks = [c.text for c in chunk_objects]
            except Exception as e:
                self.logger.error("Failed to chunk content: %s", str(e))
                # Fallback to full content if chunking fails
                chunks = [content]
        else:
            chunks = [content]

        chunk_count = len(chunks)

        # Process each chunk
        for chunk_index, chunk_text in enumerate(chunks):
            full_item_id = f"{item_id}:{chunk_index}" if chunk_count > 1 else item_id

            # 1. Generate embedding (conditionally)
            embedding_bytes = None
            if self.settings.embeddings and embed:
                if self.settings.embedding_model is None:
                    self.logger.warning(
                        "Attempted to embed content but no embedding model is configured. Skipping embedding."
                    )
                else:
                    embedding_response = await self.settings.embedding_model.arun(
                        chunk_text
                    )
                    embedding_list = embedding_response.data[0].embedding

                    # Check embedding dimensions on first chunk
                    if chunk_index == 0:
                        current_dims = self._get_embedding_dimensions()
                        if current_dims is None:
                            # No embeddings yet
                            if len(embedding_list) != self.settings.dimensions:
                                self.logger.warning(
                                    f"Embedding dimensions mismatch: expected {self.settings.dimensions}, "
                                    f"got {len(embedding_list)}. Updating dimensions to {len(embedding_list)}."
                                )
                                self.settings.dimensions = len(embedding_list)
                                # Recreate the vector table with correct dimensions
                                self.connection.execute(
                                    "DROP TABLE IF EXISTS memory_vectors"
                                )
                                self.connection.execute(f"""
                                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                                        embedding float[{self.settings.dimensions}],
                                        +item_id TEXT,
                                        +chunk_index INTEGER
                                    )
                                """)
                        elif len(embedding_list) != current_dims:
                            raise ValueError(
                                f"Embedding dimensions mismatch: existing embeddings have {current_dims} dimensions, "
                                f"but new embedding has {len(embedding_list)} dimensions."
                            )

                    embedding_bytes = serialize_float32_vector_to_bytes(embedding_list)

            # 2. Store in main table (always)
            self.connection.execute(
                """
                INSERT OR REPLACE INTO memory_items
                (id, content, metadata, chunk_index, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    full_item_id,
                    chunk_text,
                    json.dumps(metadata or {}),
                    chunk_index,
                    chunk_count,
                    datetime.now().isoformat(),
                ),
            )

            # 3. Store in FTS for keyword search (always)
            self.connection.execute(
                """
                INSERT INTO memory_fts (item_id, content)
                VALUES (?, ?)
            """,
                (
                    full_item_id,
                    chunk_text,
                ),
            )

            # 4. Store vector (conditionally)
            if embedding_bytes:
                # This block only runs if embeddings are enabled, requested, and generated
                cursor = self.connection.execute("SELECT last_insert_rowid()")
                rowid = cursor.fetchone()[0]

                self.connection.execute(
                    """
                    INSERT INTO memory_vectors (rowid, embedding, item_id, chunk_index)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        rowid,
                        embedding_bytes,
                        full_item_id,
                        chunk_index,
                    ),
                )

        self.connection.commit()
        return item_id

    async def async_search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
    ) -> list[MemorySearchResult]:
        """
        Asynchronously search memory for similar content.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            filter: Optional metadata filter (dict of key-value pairs).
            hybrid: Use hybrid search (vector + keyword). Ignored if embeddings=False.
            hybrid_weight: Weight for vector search (0=keyword, 1=vector).

        Returns:
            List of SearchResult objects.
        """

        # --- Routing Logic ---

        if not self.settings.embeddings:
            # 1. Embeddings are OFF: Force keyword-only search
            if hybrid:
                self.logger.debug(
                    "Hybrid search requested but embeddings are disabled. Falling back to keyword search."
                )
            return await self._keyword_search(query, top_k, filter)

        if hybrid:
            # 2. Embeddings are ON and Hybrid is ON
            return await self._hybrid_search(query, top_k, filter, hybrid_weight)

        # 3. Embeddings are ON and Hybrid is OFF
        return await self._vector_search(query, top_k, filter)

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filter: Dict[str, Any] | None,
    ) -> list[MemorySearchResult]:
        """Performs a pure vector similarity search."""

        # 1. Generate query embedding
        if self.settings.embedding_model is None:
            self.logger.error(
                "Vector search failed: embeddings are enabled but no model is loaded."
            )
            return []

        embedding_response = await self.settings.embedding_model.arun(query)
        query_embedding = embedding_response.data[0].embedding

        # Check query embedding dimensions against existing embeddings
        current_dims = self._get_embedding_dimensions()
        if current_dims is not None and len(query_embedding) != current_dims:
            raise ValueError(
                f"Query embedding dimensions mismatch: existing embeddings have {current_dims} dimensions, "
                f"but query embedding has {len(query_embedding)} dimensions."
            )

        query_bytes = serialize_float32_vector_to_bytes(query_embedding)

        knn_k = top_k * 5 if filter else top_k

        filter_clause = ""
        filter_values = []

        if filter:
            conditions = []
            for key, value in filter.items():
                # Note: 'm.' alias for the outer query's memory_items table
                conditions.append(f"json_extract(m.metadata, '$.{key}') = ?")
                filter_values.append(value)
            # Prepend with AND
            filter_clause = " AND " + " AND ".join(conditions)

        query_sql = f"""
            WITH knn AS (
                -- 1. Perform the raw KNN search using MATCH ? and a literal LIMIT
                -- This syntax is compatible with subqueries.
                SELECT
                    item_id,
                    distance
                FROM memory_vectors
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT {knn_k}
            )
            -- 2. Join KNN results with metadata, apply filters, and take the final top_k
            SELECT
                m.id,
                m.content,
                m.metadata,
                m.chunk_index,
                m.chunk_count,
                m.created_at,
                knn.distance
            FROM knn
            JOIN memory_items m ON knn.item_id = m.id
            WHERE 1=1 {filter_clause}
            ORDER BY knn.distance
            LIMIT {top_k}
        """

        params = [query_bytes] + filter_values
        cursor = self.connection.execute(query_sql, params)

        results = []
        for row in cursor.fetchall():
            item = MemoryItem(
                id=row[0],
                content=row[1],
                metadata=json.loads(row[2]),
                chunk_index=row[3],
                chunk_count=row[4],
                created_at=datetime.fromisoformat(row[5]),
            )
            distance = row[6]
            score = (
                1 - (distance / 2)
                if self.settings.distance_metric == "cosine"
                else 1 / (1 + distance)
            )
            results.append(
                MemorySearchResult(
                    item=item,
                    score=score,
                    distance=distance,
                    model=self.settings.embedding_model.model,
                )
            )

        return results

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filter: Dict[str, Any] | None,
    ) -> list[MemorySearchResult]:
        """Performs a pure FTS5 (keyword) search."""

        # We must use two queries to work around potential sqlite3/FTS5
        # bugs where JOINs and parameter binding fail.

        # 1. First query: Get the top candidate IDs from FTS
        fts_k = top_k * 5 if filter else top_k

        # [THE REAL, FINAL FIX]
        # The query "what do i like?" contains a literal '?'
        # which FTS5 interprets as a syntax error.
        #
        # The fix is to pass the *entire* query as a single,
        # double-quoted FTS5 phrase. This tells FTS5 to
        # "search for this exact string" and ignore any special chars.

        # We escape any internal double-quotes (to be safe)
        # and wrap the whole thing in double-quotes.
        fts_query_phrase = f'"{query.replace('"', '""')}"'

        fts_query_sql = f"""
            SELECT item_id, rank
            FROM memory_fts
            WHERE memory_fts MATCH ?
            ORDER BY rank
            LIMIT {fts_k}
        """

        try:
            # We bind the new, escaped phrase parameter
            fts_cursor = self.connection.execute(fts_query_sql, [fts_query_phrase])
            candidate_rows = fts_cursor.fetchall()
        except Exception as e:
            self.logger.error(f"FTS5 search query failed: {e}")
            self.logger.error(
                f"Failed query: {fts_query_sql} with param: {fts_query_phrase}"
            )
            return []

        if not candidate_rows:
            return []

        candidate_ids = [row[0] for row in candidate_rows]
        candidate_ranks = {row[0]: row[1] for row in candidate_rows}

        # 2. Second query: Get the full items, apply metadata filters
        placeholders = ",".join("?" * len(candidate_ids))

        filter_clause = ""
        filter_values = []

        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                filter_values.append(value)

            if conditions:
                filter_clause = " AND " + " AND ".join(conditions)

        query_sql = f"""
            SELECT
                id,
                content,
                metadata,
                chunk_index,
                chunk_count,
                created_at
            FROM memory_items
            WHERE id IN ({placeholders})
            {filter_clause}
        """

        params = candidate_ids + filter_values
        cursor = self.connection.execute(query_sql, params)

        # 3. Parse, re-sort by original rank, and take top_k
        filtered_items = []
        for row in cursor.fetchall():
            item_id = row[0]
            item = MemoryItem(
                id=item_id,
                content=row[1],
                metadata=json.loads(row[2]),
                chunk_index=row[3],
                chunk_count=row[4],
                created_at=datetime.fromisoformat(row[5]),
            )

            rank = candidate_ranks.get(item_id)
            if rank is None:
                continue

            score = 1.0 / (1.0 - rank)
            filtered_items.append(
                (rank, MemorySearchResult(item=item, score=score, distance=None))
            )

        filtered_items.sort(key=lambda x: x[0])
        results = [res for rank, res in filtered_items[:top_k]]

        return results

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filter: Dict[str, Any] | None,
        vector_weight: float,
    ) -> list[MemorySearchResult]:
        """
        Hybrid search combining vector similarity and keyword matching using RRF.
        """
        # Search for more results to increase RRF diversity
        rrf_k = 60  # RRF constant
        search_limit = top_k * 5

        # 1. Get vector results
        vector_results = await self._vector_search(
            query, top_k=search_limit, filter=filter
        )

        # 2. Get keyword (FTS5) results
        keyword_search_results = await self._keyword_search(
            query, top_k=search_limit, filter=filter
        )

        keyword_rank: Dict[str, int] = {}
        for rank, result in enumerate(keyword_search_results, 1):
            keyword_rank[result.item.id] = rank

        # 3. Reciprocal Rank Fusion (RRF)
        combined_scores: Dict[str, tuple[MemoryItem, float]] = {}

        # Add vector scores
        for rank, result in enumerate(vector_results, 1):
            item_id = result.item.id
            item = result.item
            rrf_score = vector_weight * (1 / (rank + rrf_k))
            combined_scores[item_id] = (item, rrf_score)

        # Add keyword scores
        for result in keyword_search_results:
            item_id = result.item.id
            item = result.item
            rank = keyword_rank.get(
                item_id
            )  # Get rank, may not be present if filtered out
            if rank:
                rrf_score = (1 - vector_weight) * (1 / (rank + rrf_k))
                if item_id in combined_scores:
                    existing_item, existing_score = combined_scores[item_id]
                    combined_scores[item_id] = (
                        existing_item,
                        existing_score + rrf_score,
                    )
                else:
                    combined_scores[item_id] = (item, rrf_score)

        # 4. Sort by combined score and truncate
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1][1], reverse=True
        )[:top_k]

        # 5. Convert to MemorySearchResult
        results = []
        for item_id, (item, score) in sorted_results:
            results.append(
                MemorySearchResult(
                    item=item,
                    score=score,
                    distance=1 - score,
                    model=self.settings.embedding_model.model,
                )
            )

        return results

    # --- Async Management Methods ---

    async def async_get(self, item_id: str) -> MemoryItem | None:
        """Asynchronously get item by ID."""
        cursor = self.connection.execute(
            """
            SELECT id, content, metadata, chunk_index, chunk_count, created_at
            FROM memory_items
            WHERE id = ?
        """,
            (item_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return MemoryItem(
            id=row[0],
            content=row[1],
            metadata=json.loads(row[2]),
            chunk_index=row[3],
            chunk_count=row[4],
            created_at=datetime.fromisoformat(row[5]),
        )

    async def async_delete(self, item_id: str) -> bool:
        """Asynchronously delete item and all its chunks by ID."""
        # Use LIKE to catch all chunks (e.g., "my_id:0", "my_id:1")
        search_id = f"{item_id}%"

        # Delete from all tables
        self.connection.execute(
            "DELETE FROM memory_items WHERE id LIKE ?", (search_id,)
        )
        self.connection.execute(
            "DELETE FROM memory_fts WHERE item_id LIKE ?", (search_id,)
        )
        if self.settings.embeddings:
            self.connection.execute(
                "DELETE FROM memory_vectors WHERE item_id LIKE ?", (search_id,)
            )

        self.connection.commit()
        return self.connection.total_changes > 0

    async def async_clear(self):
        """Asynchronously clear all memories."""
        self.connection.execute("DELETE FROM memory_items")
        self.connection.execute("DELETE FROM memory_fts")
        if self.settings.embeddings:
            self.connection.execute("DELETE FROM memory_vectors")
        self.connection.commit()

    async def async_rag(
        self,
        query: str,
        language_model: LanguageModelName | str | LanguageModel,
        *,
        type: Type[T_ResponseModel] = str,
        prompt_template: str | None = None,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
        hybrid: bool = True,
        hybrid_weight: float = 0.5,
        **kwargs,
    ) -> MemoryQueryResponse[T_ResponseModel]:
        """
        Asynchronously search memory and then query a language model with the results.

        Args:
            query: The user's query to search for and answer.
            language_model: An initialized `zyx.models.LanguageModel` instance.
            type: The expected Pydantic model for a structured response,
                  or `str` (default) for a text response.
            prompt_template: An optional f-string template for the prompt.
                             Must include `{context}` and `{query}` placeholders.
            top_k: Number of search results to retrieve.
            filter: Metadata filter for the search.
            hybrid: Whether to use hybrid search (if embeddings are enabled).
            hybrid_weight: The weight for vector search in hybrid mode.
            **kwargs: Additional keyword arguments to pass to `language_model.arun()`.

        Returns:
            A `MemoryQueryResponse` object.
        """
        if not isinstance(language_model, LanguageModel):
            try:
                language_model = LanguageModel(language_model, adapter="auto")
            except Exception as e:
                self.logger.error(
                    "Failed to initialize LanguageModel for RAG: %s", str(e)
                )
                raise

        # 1. Perform the search
        self.logger.debug(f"Querying memory for: '{query}'")
        search_results = await self.async_search(
            query,
            top_k=top_k,
            filter=filter,
            hybrid=hybrid
            if self.settings.embeddings
            else False,  # Hybrid only if embeddings are on
            hybrid_weight=hybrid_weight,
        )

        # 2. Format the context
        if not search_results:
            context_str = "No relevant context was found in memory."
            self.logger.debug("No context found for query.")
        else:
            context_str = "\n\n".join(
                [f"Context Item (ID: {r.id}):\n{r.content}" for r in search_results]
            )
            self.logger.debug(f"Retrieved {len(search_results)} items for context.")

        # 3. Select and format the prompt
        if prompt_template is None:
            prompt_template = DEFAULT_QUERY_PROMPT

        try:
            formatted_prompt = prompt_template.format(context=context_str, query=query)
        except KeyError as e:
            self.logger.error(
                f"Prompt template formatting failed. Ensure it includes '{{context}}' and '{{query}}'. Error: {e}"
            )
            raise ValueError(
                f"Invalid prompt_template. Missing key: {e}. "
                "Template must include {context} and {query}."
            )

        # 4. Call the language model
        self.logger.debug(f"Sending prompt to language model: {language_model.model}")

        # Ensure streaming is disabled, as this method returns a single response
        if "stream" in kwargs and kwargs["stream"]:
            self.logger.warning(
                "Memory.async_rag() does not support streaming. Forcing 'stream=False'."
            )
        kwargs["stream"] = False

        llm_response: LanguageModelResponse[
            T_ResponseModel
        ] = await language_model.arun(formatted_prompt, type=type, **kwargs)
        self.logger.debug("Received response from language model.")

        # 5. Package and return the final response
        return MemoryQueryResponse[T_ResponseModel](
            response=llm_response, results=search_results
        )

    # --- Other Methods ---

    def count(self) -> int:
        """Get total number of unique original memory items (not chunks)."""
        cursor = self.connection.execute("""
            SELECT COUNT(DISTINCT 
                CASE 
                    WHEN chunk_count > 1 
                    THEN SUBSTR(id, 1, INSTR(id, ':') - 1) 
                    ELSE id 
                END
            ) FROM memory_items
        """)
        return cursor.fetchone()[0]

    def close(self):
        """Close database connection."""
        if self._connection:
            self.logger.debug("Closing memory database connection.")
            self._connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
