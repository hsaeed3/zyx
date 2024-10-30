from __future__ import annotations as __annotations__

__annotations__
"""
zyx // super duper simple llm framework
"""

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

from .lib.router import router


# ==============================
# UTILS
# ==============================


from .lib.utils import logger, console


# ==============================
# DATA
# ==============================



# ==============================
# Completions & METHODS
# ==============================


from .completions.base_client import Client as _Client, completion


from .data.basemodel import BaseModel
from pydantic import Field


class Completions(router):
    pass

Completions.init("zyx.completions.main", "Completions")


class classifier(router):
    pass

classifier.init("zyx.completions.methods.classifier", "classifier")


class coder(router):
    pass

coder.init("zyx.completions.methods.code_constructor", "coder")


class extractor(router):
    pass

extractor.init("zyx.completions.methods.extractor", "extractor")


class function(router):
    pass

function.init("zyx.completions.methods.function_constructor", "function")


class generator(router):
    pass

generator.init("zyx.completions.methods.generator", "generator")


class patcher(router):
    pass

patcher.init("zyx.completions.methods.patcher", "patcher")


class planner(router):
    pass

planner.init("zyx.completions.methods.planning", "planner")


class query(router):
    pass

query.init("zyx.completions.methods.question_answer", "query")


class selector(router):
    pass

selector.init("zyx.completions.methods.selector", "selector")


class solver(router):
    pass

solver.init("zyx.completions.methods.solver", "solver")


class validator(router):
    pass

validator.init("zyx.completions.methods.validator", "validator")