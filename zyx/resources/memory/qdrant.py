"""zyx.resources.memory.qdrant"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import warnings

try:
    from qdrant_client import AsyncQdrantClient  # type: ignore[unresolved-import]
    from qdrant_client.http import models  # type: ignore[unresolved-import]
except ImportError:
    raise ImportError(
        "To use the `qdrant` memory provider, you must first install the `qdrant-client` library.\n"
        "You can install it by using one of the following commands:\n"
        "```bash\n"
        "pip install zyx[qdrant]\n"
        "pip install 'qdrant-client[fastembed]'\n"
        "```"
    )

from . import MemoryProvider

warnings.filterwarnings(
    "ignore",
    message=r".*`add` method has been deprecated.*",
    category=UserWarning,
    module=r".*qdrant_client\.common\.client_warnings.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*`query` method has been deprecated.*",
    category=UserWarning,
    module=r".*qdrant_client\.common\.client_warnings.*",
)

__all__ = ("QdrantMemoryProvider",)


@dataclass(init=False)
class QdrantMemoryProvider(MemoryProvider):
    """
    Memory provider that utilizes `qdrant-client` and `fastembed` for
    embeddings. Uses a local persisted store by default.
    """

    @property
    def client(self) -> AsyncQdrantClient | None:
        """The `qdrant_client.AsyncQdrantClient` instance for this memory provider."""
        return getattr(self, "_client", None)

    @property
    def path(self) -> str | Path | None:
        """The path to the Qdrant database for this memory provider."""
        if self.client is None:
            return None
        return getattr(self.client, "path", None) or getattr(
            self.client, "location", None
        )

    @property
    def collection_name(self) -> str:
        """The name of the collection to use. May contain a `{key}` placeholder."""
        return getattr(self, "_collection_name", "memory-{key}")

    @collection_name.setter
    def collection_name(self, value: str) -> None:
        self._collection_name = value

    def __init__(
        self,
        client: AsyncQdrantClient | Path | str | None = None,
        path: str | Path | None = None,
        collection_name: str = "memory-{key}",
    ):
        """
        Initialize a new `QdrantMemoryProvider` instance.

        Args:
            client : `AsyncQdrantClient` | `Path` | `str` | None
                An existing `AsyncQdrantClient`, or a path for local persistence.
                If None, uses `path` or default `Path.cwd() / "memory/qdrant"`.
            path : `str` | `Path` | None
                Directory path for local Qdrant storage. Used only when `client`
                is None. Defaults to current working directory + `memory/qdrant`.
            collection_name : `str`
                Collection name template; `{key}` is replaced by the memory key.
        """
        if client is not None and not isinstance(client, AsyncQdrantClient):
            path = Path(client) if isinstance(client, str) else client
        if path is None:
            path = Path.cwd() / "memory/qdrant"
        if not isinstance(client, AsyncQdrantClient):
            self._client = AsyncQdrantClient(path=str(path))
        else:
            self._client = client
        self._collection_name = collection_name

    async def get_collection(self, key: str) -> models.CollectionInfo:
        """Return collection info for the given memory key."""
        name = self.collection_name.format(key=key)

        if self.client is not None and hasattr(self.client, "get_collection"):
            return await self.client.get_collection(collection_name=name)

    async def add(self, key: str, content: str) -> str:
        memory_id = str(uuid.uuid4())
        collection_name = self.collection_name.format(key=key)

        if self.client is not None and hasattr(self.client, "add"):
            ids = await self.client.add(
                collection_name=collection_name,
                documents=[content],
                metadata=[{"id": memory_id}],
                ids=[memory_id],
            )
        return str(ids[0]) if ids else memory_id

    async def delete(self, key: str, id: str) -> None:
        collection_name = self.collection_name.format(key=key)

        if self.client is not None and hasattr(self.client, "delete"):
            await self.client.delete(
                collection_name=collection_name,
                points_selector=[id],
            )

    async def search(
        self, key: str, query: str, n: int = 20
    ) -> Dict[str, str]:
        collection_name = self.collection_name.format(key=key)

        if self.client is not None and hasattr(self.client, "query"):
            results = await self.client.query(
                collection_name=collection_name,
                query_text=query,
                limit=n,
            )
        return {str(r.id): r.document for r in results}
