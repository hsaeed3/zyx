# zyx ==============================================================================

from ._loader import UtilLazyLoader


class db(UtilLazyLoader):
    pass


db.init("sqlmodel.main", "SQLModel")


class memory(UtilLazyLoader):
    pass


memory.init("mem0.memory.main", "Memory")


class qdrant(UtilLazyLoader):
    pass


qdrant.init("qdrant_client.qdrant_client", "QdrantClient")
