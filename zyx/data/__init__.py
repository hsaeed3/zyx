__all__ = [
    "BaseModel",
    "Field",
    "chunk",
    "embeddings",
    "Memory",
    "read",
    "read_url",
    "scrape",
    "web_search"
]


from ..lib.router import router

from .basemodel import BaseModel
from pydantic import Field


class chunk(router):
    pass

chunk.init("zyx.data.chunker", "chunk")


class embeddings(router):
    pass

embeddings.init("zyx.data.embedder", "embeddings")


class Memory(router):
    pass

Memory.init("zyx.data.memory", "Memory")


class read(router):
    pass

read.init("zyx.data.reader", "read")


class read_url(router):
    pass

read_url.init("zyx.data.url_reader", "read_url")


class scrape(router):
    pass

scrape.init("zyx.data.scraper", "scrape")


class web_search(router):
    pass

web_search.init("zyx.data.web_search", "web_search")
    