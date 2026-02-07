"""zyx.types"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Type,
    TypeAlias,
    TypeVar,
    Union
)

from pydantic import BaseModel
from pydantic_ai.agent import Agent
from pydantic_ai import (
    messages as _pydantic_ai_messages,
    models as _pydantic_ai_models,
    tools as _pydantic_ai_tools,
    toolsets as _pydantic_ai_toolsets,
)
from pydantic_ai.agent.abstract import (
    Instructions as InstructionsParam,
)

__all__ = (
    "Output",
    "DepsType",
    "TargetParam",
    "ModelParam",
    "ContextParam",
    "ToolParam",
    "InstructionsParam",
)


Output = TypeVar("Output")
"""Alias for the final output of a semantic operation."""


DepsType = TypeVar("DepsType")
"""Alias for `deps` within `pydantic_ai`'s RunContext"""


TargetParam : TypeAlias = Union[
    Output,
    Type[Output],
]
"""
The output 'target' type or value that should be generated/modified by a 
semantic operation.
"""


ModelParam : TypeAlias = Union[
    str,
    Agent[DepsType, Output],
    _pydantic_ai_models.KnownModelName,
    _pydantic_ai_models.Model,
]


ContextParam : TypeAlias = Union[
    str,
    Dict[str, Any],
    BaseModel,
    _pydantic_ai_messages.ModelMessage,
]
"""
Accepted formats in which a single piece of content within the `context` parameter
of a semantic operation can be passed.

Accepts:
- **str** (String content, that can contain role tags : [s]/[system], [u]/[user], [a]/[assistant])
- **Dict[str, Any]** / **BaseModel** (A dictionary/pydantic model in the PydanticAI ModelMessage or OpenAI Chat Completions format)
- **ModelMessage** (A PydanticAI ModelMessage object)
"""


ToolParam : TypeAlias = Union[
    Callable[..., Any],
    _pydantic_ai_tools.Tool,
    _pydantic_ai_tools.AbstractBuiltinTool,
    _pydantic_ai_toolsets.AbstractToolset,
]
"""
Unified representation of all tool types that can be passed to a semantic operation.

This includes all types `pydantic_ai` provides as tools/toolsets.
"""