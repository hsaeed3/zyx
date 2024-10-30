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

from .completions.methods.classifier import classify
from .completions.methods.code_constructor import coder
from .completions.methods.extractor import extract
from .completions.methods.function_constructor import function
from .completions.methods.generator import generate
from .completions.methods.patcher import patch
from .completions.methods.prompts import prompter
from .completions.methods.planning import planner
from .completions.methods.queries import query
from .completions.methods.selector import select
from .completions.methods.solver import solve
from .completions.methods.validator import validate

