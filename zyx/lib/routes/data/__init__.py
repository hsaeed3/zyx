__all__ = [
    "embeddings"

    "Rag",
    "Sql",
    "VectorStore"
]


from .._loader import loader


class embeddings(loader):
    pass


embeddings.init("litellm.main", "embedding")


class Rag(loader):
    pass


Rag.init("zyx.lib.data.rag_store", "Rag")


class Sql(loader):
    pass


Sql.init("zyx.lib.data.sql_store", "Sql")


class VectorStore(loader):
    pass


VectorStore.init("zyx.lib.data.vector_store", "VectorStore")
