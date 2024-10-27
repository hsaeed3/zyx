__all__ = [
    "chunk",
    "embeddings",
    "read",

    "Store", "Document",
]


from .._router import router


class Store(router):
    """
    A store for vector embeddings.
    """
    pass


Store.init("zyx.data.vector_store.store", "Store")


from .document import Document
from .resources import reader

read = reader.read

from .resources import embedder

embeddings = embedder.embeddings

from .resources import chunker

chunk = chunker.chunk
