__all__ = [
    # zyx main modules

    # utils
    "logger",
    "console",

    # base
    "_Client",

    # Completions Client
    # all zyx methods & agentic framework
    # if you want to directly use the agents module `from zyx.agents import Agents`
    "Completions",
    # singleton completion client
    "completion",

    # single-shot methods
    "classify",
    "coder",
    "extract",
    "function",
    "generate",
    "patch",
    "prompter",
    "planner",
    "query",
    "qa",
    "select",
    "solve",
    "validate",

    # data
    "BaseModel",
    "Field",
    "chunk",
    "embeddings",
    "Memory",
    "read",
    "read_url",
    "scrape",
    "web_search",
]

from .completions.base_client import Client as _Client, completion

from .completions.main import Completions

from .data import (
    BaseModel,
    Field,
    chunk,
    embeddings,
    Memory,
    read,
    read_url,
    scrape,
    web_search
)

from .lib.utils import logger, console


