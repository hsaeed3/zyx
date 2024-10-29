__all__ = [
    "Memory",
    "Document",
    "embeddings",
    "scrape",
    "chunk",
    "read",
]


from .._router import router


class Memory(router):
    pass


Memory.init("zyx.resources.stores.memory", "Memory")


class Document(router):
    pass


Document.init("zyx.lib.types.document", "Document")


class embeddings(router):
    pass


embeddings.init("litellm.main", "embedding")


class scrape(router):
    pass


scrape.init("zyx.resources.completions.agents.scrape", "scrape")


class chunk(router):
    pass


chunk.init("zyx.resources.data.chunk", "chunk")


class read(router):
    pass


read.init("zyx.resources.data.reader", "read")
