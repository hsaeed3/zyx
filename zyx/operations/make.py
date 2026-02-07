"""zyx.operations.make"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

from pydantic_ai.agent import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ..exceptions import (
    InvalidTargetError,
    AgentRunError,
)
from ..utils import core
from ..processing.outputs import normalize_output_type
from ..results import Result
from ..types import (
    DepsType,
    Output,
    InstructionsParam,
    ModelParam,
    ContextParam,
    TargetParam,
    ToolParam,
)

__all__ = (
    "make",
    "amake",
)


def _prepare_make_request(params: Dict[str, Any]) -> Tuple[Agent, Dict[str, Any]]:
    """
    Takes in a dictionary of parameters used to invoke the `make` or `amake` operations,
    and returns tuple containing a `pydantic_ai.Agent` object and
    a dictionary of parameters compatible with the agent's run or run_stream methods.
    """
    request_params: Dict[str, Any] = {}

    model = params.get("model", None)
    model_settings = params.get("model_settings", None)

    agent = core.prepare_agent(
        model=model,
        model_settings=model_settings,
        deps_type=type(params.get("deps", None))
        if params.get("deps", None)
        else None,
    )

    request_params = core.prepare_context(
        context=params.get("context", None),
        instructions=params.get("instructions", None),
    )

    parsed_tools = core.prepare_tools(params.get("tools", None))
    if parsed_tools.get("toolsets", None):
        request_params["toolsets"] = parsed_tools.get("toolsets")
    if parsed_tools.get("builtin_tools", None):
        request_params["builtin_tools"] = parsed_tools.get("builtin_tools")

    if params.get("deps", None):
        request_params["deps"] = params.get("deps")

    if params.get("usage_limits", None):
        request_params["usage_limits"] = params.get("usage_limits")

    try:
        request_params["output_type"] = normalize_output_type(
            params.get("target", None)
        )
    except Exception as e:
        raise InvalidTargetError(
            "Recieved an invalid target type/object to generate a result for.\n"
            f"The target type/object {params.get('target', None).__name__} is not yet supported."  # type: ignore
        ) from e

    return (agent, request_params)


def make(
    context: ContextParam | List[ContextParam],
    target: TargetParam = str,
    *,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: ModelSettings | None = None,
    instructions: InstructionsParam | None = None,
    tools: ToolParam | List[ToolParam] | None = None,
    deps: DepsType | None = None,
    usage_limits: UsageLimits | None = None,
    **kwargs: Any,
) -> Result[Output]:
    """Generate ('make') a result in a `target` type using an LLM or Agent.

    Examples:

        from zyx import make

        response = make(
            context = "What is 2+2?",
            target = int
        )

        print(response.output)

    Args:
        context (ContextParam | List[ContextParam]): The context to use for the semantic operation.
        target (TargetParam, optional): The target type to generate a result for. Defaults to str.
        model (ModelParam, optional): The model to use for the semantic operation. Defaults to "openai:gpt-4o-mini".
        model_settings (ModelSettings | None, optional): The model settings to use for the semantic operation. Defaults to None.
        instructions (InstructionsParam | None, optional): The instructions to use for the semantic operation. Defaults to None.
        tools (ToolParam | List[ToolParam] | None, optional): The tools to use for the semantic operation. Defaults to None.
        deps (DepsType | None, optional): The dependencies to use for the semantic operation. Defaults to None.
        usage_limits (UsageLimits | None, optional): The usage limits to use for the semantic operation. Defaults to None.
        **kwargs (Any): Additional keyword arguments to pass to the semantic operation.

    Returns:
        Result[Output]: A `Result` object containing the output of the semantic operation.

    Raises:
        InvalidTargetError: If the target type is not supported.
        AgentRunError: If an error occurs while running the agent.
    """
    params = {
        "context": context,
        "target": target,
        "model": model,
        "model_settings": model_settings,
        "instructions": instructions,
        "tools": tools,
        "usage_limits": usage_limits,
        "deps": deps,
        **kwargs,
    }

    agent, request_params = _prepare_make_request(params=params)

    try:
        result = agent.run_sync(**request_params)
    except Exception as e:
        raise AgentRunError(operation_kind="make", agent=agent, error=e) from e

    return Result(
        kind="make",
        output=result.output,
        raw=result,
        models=[
            agent.model if isinstance(agent.model, str) else agent.model.model_name
        ],  # type: ignore
    )


async def amake(
    context: ContextParam | List[ContextParam],
    target: TargetParam = str,
    *,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: ModelSettings | None = None,
    instructions: InstructionsParam | None = None,
    tools: ToolParam | List[ToolParam] | None = None,
    deps: DepsType | None = None,
    usage_limits: UsageLimits | None = None,
    **kwargs: Any,
) -> Result[Output]:
    """Asynchronously generate ('make') a result in a `target` type using an LLM or Agent.

    Examples:

        import asyncio
        from zyx import amake

        async def example():
            response = await amake(
                context = "What is 2+2?",
                target = int
            )
            print(response.output)

        asyncio.run(example())

    Args:
        context (ContextParam | List[ContextParam]): The context to use for the semantic operation.
        target (TargetParam, optional): The target type to generate a result for. Defaults to str.
        model (ModelParam, optional): The model to use for the semantic operation. Defaults to "openai:gpt-4o-mini".
        model_settings (ModelSettings | None, optional): The model settings to use for the semantic operation. Defaults to None.
        instructions (InstructionsParam | None, optional): The instructions to use for the semantic operation. Defaults to None.
        tools (ToolParam | List[ToolParam] | None, optional): The tools to use for the semantic operation. Defaults to None.
        deps (DepsType | None, optional): The dependencies to use for the semantic operation. Defaults to None.
        usage_limits (UsageLimits | None, optional): The usage limits to use for the semantic operation. Defaults to None.
        **kwargs (Any): Additional keyword arguments to pass to the semantic operation.

    Returns:
        Result[Output]: A `Result` object containing the output of the semantic operation.

    Raises:
        InvalidTargetError: If the target type is not supported.
        AgentRunError: If an error occurs while running the agent.
    """

    params = {
        "context": context,
        "target": target,
        "model": model,
        "model_settings": model_settings,
        "instructions": instructions,
        "tools": tools,
        "usage_limits": usage_limits,
        "deps": deps,
    }

    agent, request_params = _prepare_make_request(params=params)

    try:
        result = await agent.run(
            **request_params,
            **kwargs,
        )
    except Exception as e:
        raise AgentRunError(operation_kind="make", agent=agent, error=e) from e

    return Result(
        kind="make",
        output=result.output,
        raw=result,
        models=[
            agent.model if isinstance(agent.model, str) else agent.model.model_name
        ],  # type: ignore
    )
