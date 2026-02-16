"""zyx._aliases

Provides interface-level aliases that re-export various components and types from the
`pydantic_ai` library for more coherent usage throughout `zyx`.
"""

from __future__ import annotations

from typing import TypeAlias

from pydantic_ai import (
    _function_schema as _pydantic_ai_function_schema,
    agent as _pydantic_ai_agent,
    messages as _pydantic_ai_messages,
    models as _pydantic_ai_models,
    result as _pydantic_ai_result,
    run as _pydantic_ai_run,
    tools as _pydantic_ai_tools,
    toolsets as _pydantic_ai_toolsets,
    settings as _pydantic_ai_settings,
    usage as _pydantic_ai_usage,
    RunContext as PydanticAIRunContext,
)

__all__ = (
    "PydanticAIRunContext",
    "PydanticAIAgent",
    "PydanticAIAgentResult",
    "PydanticAIAgentStream",
    "PydanticAIInstructions",
    "PydanticAIMessage",
    "PydanticAIModel",
    "PydanticAIModelName",
    "PydanticAIModelSettings",
    "PydanticAIModelRequest",
    "PydanticAIModelResponse",
    "PydanticAISystemPromptPart",
    "PydanticAIUserPromptPart",
    "PydanticAIUserContent",
    "PydanticAITool",
    "PydanticAIBuiltinTool",
    "PydanticAIToolset",
    "PydanticAIUsage",
    "PydanticAIUsageLimits",
    "_pydantic_ai_function_schema",
)


PydanticAIAgent: TypeAlias = _pydantic_ai_agent.Agent


PydanticAIAgentResult: TypeAlias = _pydantic_ai_run.AgentRunResult


PydanticAIAgentStream: TypeAlias = _pydantic_ai_result.StreamedRunResult


PydanticAIInstructions: TypeAlias = _pydantic_ai_agent.Instructions


PydanticAIMessage: TypeAlias = _pydantic_ai_messages.ModelMessage


PydanticAIModel: TypeAlias = _pydantic_ai_models.Model


PydanticAIModelName: TypeAlias = _pydantic_ai_models.KnownModelName


PydanticAIModelSettings: TypeAlias = _pydantic_ai_settings.ModelSettings


PydanticAIModelRequest: TypeAlias = _pydantic_ai_messages.ModelRequest


PydanticAIModelResponse: TypeAlias = _pydantic_ai_messages.ModelResponse


PydanticAISystemPromptPart: TypeAlias = _pydantic_ai_messages.SystemPromptPart


PydanticAIUserPromptPart: TypeAlias = _pydantic_ai_messages.UserPromptPart


PydanticAIUserContent: TypeAlias = _pydantic_ai_messages.UserContent


PydanticAITool: TypeAlias = _pydantic_ai_tools.Tool


PydanticAIBuiltinTool: TypeAlias = _pydantic_ai_tools.AbstractBuiltinTool


PydanticAIToolset: TypeAlias = _pydantic_ai_toolsets.AbstractToolset


PydanticAIUsage: TypeAlias = _pydantic_ai_usage.RunUsage


PydanticAIUsageLimits: TypeAlias = _pydantic_ai_usage.UsageLimits
