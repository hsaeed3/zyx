from __future__ import annotations as __annotations__

__all__ = [
    "_Client",
    "Completions",
    "completion",

    "BaseModel", "Field",

    "chunk", "read", "embeddings",

    "image", "audio", "transcribe",

    "change_default_model", "reset_config",

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

    "console",
    "logger",

    "Store", "Document",

    "ZYX_DEFAULT_MODEL",
]

from .base_client import Client as _Client

# -- util --

from ._rich import console, logger

# -- env --

from ._environ import Environment as __environ, ZYX_DEFAULT_MODEL

change_default_model = __environ.change_default_model
reset_config = __environ.reset_config

# -- basemodel --

from .basemodel import BaseModel, Field


# -- multimodal --

from .multimodal import image, audio, transcribe

# -- data --

from .data.resources import chunker as __chunker, reader as __reader, embedder as __embedder


# -- completions client --

from .completions import (
    Completions,
    classifier as __classifier, coder as __coder, extractor as __extractor, function_constructor as __function_constructor, generator as __generator,
    patcher as __patcher, planner as __planner, prompts as __prompts, queries as __queries, question_answer as __question_answer, selector as __selector,
    solver as __solver, validator as __validator,
    completion
)

# -- vector store --
# only compatible with zyx[all]

from .data import Store, Document

# -- functions --

classify = __classifier.classify
coder = __coder.coder
extract = __extractor.extract
function = __function_constructor.function
generate = __generator.generate
patch = __patcher.patch
planner = __planner.planner
prompter = __prompts.prompter
query = __queries.query
qa = __question_answer.qa
select = __selector.select
solve = __solver.solve
validate = __validator.validate

chunk = __chunker.chunk
read = __reader.read
embeddings = __embedder.embeddings
