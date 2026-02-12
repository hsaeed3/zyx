from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, TypeAlias

try:
    import chromadb # type: ignore[import-untyped]
except ImportError:
    raise ImportError(
        f"To use the `chroma` memory provider, you must first install the `chromadb` library.\n"
        "You can install it by using one of the following commands:\n"
        "```bash\n"
        "pip install zyx[chroma]\n"
        "pip install chromadb\n"
        "```"
    )

from . import MemoryProvider


ChromaMemoryClient: TypeAlias = Literal["persistent", "ephemeral"]


@dataclass(init=False)
class ChromaMemoryProvider(MemoryProvider):
    """
    Memory provider that utilizes `chromadb`.
    """

    @property
    def client(self) -> chromadb.ClientAPI:
        """The `chromadb.ClientAPI` instance to use for this memory provider."""
        return getattr(self, "_client", None)

    @property
    def path(self) -> str | Path | None:
        """An explicit directory path to use for the persistent client, if not
        provided this will use the current working directory + `memory/chroma`.
        """
        return getattr(self, "_path", None)

    @property
    def collection_name(self) -> str:
        """An optional name for the collection to use for this memory provider.

        This can contain the `{key}` placeholder, which will inject the name
        of the `Memory` resource using this provider."""
        return getattr(self, "_collection_name", "memory-{key}")

    @collection_name.setter
    def collection_name(self, value: str) -> None:
        self._collection_name = value

    def __init__(
        self,
        client: chromadb.ClientAPI | ChromaMemoryClient = "persistent",
        path: str | Path | None = None,
        collection_name: str = "memory-{key}",
    ):
        """
        Initialize a new `ChromaMemoryProvider` instance.

        Args:
            client : `chromadb.ClientAPI` | `ChromaMemoryClient`
                The `chromadb.ClientAPI` instance to use for this memory provider, or the type of client to use.
            path : `str` | `Path` | None
                An explicit directory path to use for the persistent client, if not
                provided this will use the current working directory + `memory/chroma`.
            collection_name : `str`
                An optional name for the collection to use for this memory provider.

                This can contain the `{key}` placeholder, which will inject the name
                of the `Memory` resource using this provider.
        """
        if client == "persistent":
            if not self.path:
                path = Path.cwd() / "memory/chroma"
            self._client = chromadb.PersistentClient(path=str(path))
        else:
            self._client = chromadb.EphemeralClient()

        if path:
            self._path = path

        self._collection_name = collection_name

    def get_collection(self, key: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(
            self.collection_name.format(key=key)
        )

    async def add(self, key: str, content: str) -> str:
        collection = self.get_collection(key)
        id = str(uuid.uuid4())

        collection.add(documents=[content], metadatas=[{"id": id}], ids=[id])
        return id

    async def delete(self, key: str, id: str) -> None:
        collection = self.get_collection(key)
        collection.delete(ids=[id])

    async def search(
        self, key: str, query: str, n: int = 20
    ) -> Dict[str, str]:
        collection = self.get_collection(key)

        results = collection.query(query_texts=[query], n_results=n)
        if not results.get("ids") or not results.get("documents"):
            return {}

        return dict(
            zip(
                results["ids"][0],
                results["documents"][0],
                strict=False,
            )
        )
