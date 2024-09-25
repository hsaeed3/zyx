__all__ = [
    "sql_store",
    "rag",
    "vector_store"
]


from ..utils._loader import loader


class sql_store(loader):
    pass


sql_store.init("zyx.lib.data.store", "Store")


class rag(loader):
    pass


rag.init("zyx.lib.data.rag_store", "RagStore")


class vector_store(loader):
    pass


vector_store.init("zyx.lib.data.vector_store", "VectorStore")

