"""zyx._aliases

Provides interface-level aliases that re-export various components and types from the
`pydantic_ai` library for more coherent usage throughout `zyx`.
"""

from __future__ import annotations

from pydantic_ai import (
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
)


PydanticAIAgent = _pydantic_ai_agent.Agent


PydanticAIAgentResult = _pydantic_ai_run.AgentRunResult


PydanticAIAgentStream = _pydantic_ai_result.StreamedRunResult


PydanticAIInstructions = _pydantic_ai_agent.Instructions


PydanticAIMessage = _pydantic_ai_messages.ModelMessage


PydanticAIModel = _pydantic_ai_models.Model


PydanticAIModelName = _pydantic_ai_models.KnownModelName


PydanticAIModelSettings = _pydantic_ai_settings.ModelSettings


PydanticAIModelRequest = _pydantic_ai_messages.ModelRequest


PydanticAIModelResponse = _pydantic_ai_messages.ModelResponse


PydanticAISystemPromptPart = _pydantic_ai_messages.SystemPromptPart


PydanticAIUserPromptPart = _pydantic_ai_messages.UserPromptPart


PydanticAIUserContent = _pydantic_ai_messages.UserContent


PydanticAITool = _pydantic_ai_tools.Tool


PydanticAIBuiltinTool = _pydantic_ai_tools.AbstractBuiltinTool


PydanticAIToolset = _pydantic_ai_toolsets.AbstractToolset


PydanticAIUsage = _pydantic_ai_usage.RunUsage


PydanticAIUsageLimits = _pydantic_ai_usage.UsageLimits
