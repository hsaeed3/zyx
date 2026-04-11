"""zyx._types

Framework specific types and aliases most commonly found within the function signatures
of semantic operations.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

from pydantic import BaseModel
from pydantic_ai import (
    agent as _pydantic_ai_agent,
    models as _pydantic_ai_models,
)

from ._aliases import (
    PydanticAIAgent,
    PydanticAIModel,
    PydanticAIMessage,
    PydanticAITool,
    PydanticAIToolset,
    PydanticAIBuiltinTool,
)
from .attachments import Attachment, AttachmentLike
from .context import Context
from .tools.memory import Memory
from .tools.code import Code

__all__ = (
    "Deps",
    "Output",
    "InstructionsParam",
    "SourceParam",
    "TargetParam",
    "ModelParam",
    "ContextType",
    "ToolType",
    "AttachmentType",
)


Deps = TypeVar("Deps")
"""Alias for `deps` used within `pydantic_ai.RunContext`."""


Output = TypeVar("Output")
"""Generic type variable for the final output found within the `Result` or
`Stream` returned by a semantic operation."""


InstructionsParam: TypeAlias = _pydantic_ai_agent.Instructions | Any
"""Accepted formats of passing in instructions to a semantic operation. This is a re-export
of the `pydantic_ai.agent.Instructions` type, and supports:

- Strings / Lists of strings
- Functions that return a string / and can optionally accept `RunContext` dependencies
"""


SourceParam: TypeAlias = Any
"""The input object to be considered the source of truth or 'main' data/context within
a semantic operation."""


TargetParam: TypeAlias = Union[Output, Type[Output]]
"""The output `target` - a type, instance, or value to generate/modify through
a semantic operation."""


KnownModelName: TypeAlias = Literal[
    "ollama:functiongemma",
    "ollama:deepseek-ocr",
    "ollama:deepseek-v3.1",
    "ollama:devstral-small-2",
    "ollama:devstral-small-2:cloud",
    "ollama:lfm2.5-thinking",
    "ollama:glm-4.7-flash",
    "ollama:glm-5:cloud",
    "ollama:glm-ocr",
    "ollama:gpt-oss",
    "ollama:gpt-oss-safeguard",
    "ollama:granite4",
    "ollama:kimi-k2.5:cloud",
    "ollama:minimax-m2.5:cloud",
    "ollama:ministral-3",
    "ollama:nemotron-3-nano",
    "ollama:olmo-3",
    "ollama:olmo-3.1",
    "ollama:translategemma",
    "ollama:rnj-1",
    "ollama:qwen3-coder-next",
    "ollama:qwen3-coder-next:cloud",
    "ollama:qwen3-next",
    "ollama:qwen3-vl",
]


ModelParam: TypeAlias = Union[
    PydanticAIAgent,
    PydanticAIModel,
    KnownModelName,
    _pydantic_ai_models.KnownModelName,
    str,
]
"""Accepted formats for a `runner` or generator that will be used to produce the output
of a semantic operation.

This can be passed as:
- A string name of a known model in the `pydantic_ai` format (e.g. "openai:gpt-4o-mini")
- A PydanticAI Model (OpenAIChatModel, AnthropicModel, etc.)
- A PydanticAI Agent
"""


ContextType: TypeAlias = Union[
    str,
    PydanticAIMessage,
    BaseModel,
    Dict[str, Any],
    Context,
    AttachmentLike,
    Attachment
]
"""Accepted formats for a single item within the ``context`` of a semantic
operation.  All items are converted to ``pydantic_ai.ModelMessage`` objects
before invocation.

Accepted types:

- A :class:`~zyx.context.Context` accumulates conversation state across
  operations.  When passed, its messages and rendered instructions are
  injected into the request, and it is auto-updated after the operation
  when ``Context.update`` is ``True``.

- A string (with optional role tags like ``[system]``/``[s]``,
  ``[user]``/``[u]``, ``[assistant]``/``[a]``)

- A dictionary or Pydantic model in the PydanticAI or OpenAI Chat Completions
  format

- A ``pydantic_ai.ModelMessage``

Examples::

    ctx = Context()
    ctx.add_user("Hi!")
    result = zyx.make(context=[ctx, "What did I just say?"], target=str)
"""


ToolType: TypeAlias = Union[
    Callable[..., Any],
    PydanticAITool,
    PydanticAIBuiltinTool,
    PydanticAIToolset,
    Memory,
    Code,
]
"""
Accepted formats in which a single item within the `tools` parameter of a semantic operation
can be passed in as. All tools support passing in RunContext dependencies.

This is a unification of all representations of a tool that are provided by `pydantic_ai`,
including:

- Functions
- PydanticAI Tools
- PydanticAI Builtin Tools
- PydanticAI Toolsets
"""


AttachmentType: TypeAlias = Union[Any, Type[Any], Attachment, AttachmentLike]
"""
Accepted formats in which a single item within the `attachments` parameter of a semantic operation
can be passed in as.
"""
