from __future__ import annotations as __annotations__

__annotations__
"""
zyx // super duper simple llm framework
"""

__all__ = [
    # zyx main modules

    # Completions Client
    # all zyx methods & agentic framework
    # if you want to directly use the agents module `from zyx.agents import Agents`
    "Completions",
    # singleton completion client
    "completion",

    # zyx BaseModel
    # pydantic extension
    "BaseModel", "Field",

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
]

# client
from .completions.main import (
    Completions, completion
)

from .completions.methods.classifier import classify
from .completions.methods.code_constructor import coder
from .completions.methods.extractor import extract
from .completions.methods.function_constructor import function
from .completions.methods.generator import generate
from .completions.methods.patcher import patch
from .completions.methods.prompts import prompter
from .completions.methods.planning import planner
from .completions.methods.queries import query
from .completions.methods.question_answer import qa
from .completions.methods.selector import select
from .completions.methods.solver import solve
from .completions.methods.validator import validate

# pydantic
from .extensions.basemodel import BaseModel, Field