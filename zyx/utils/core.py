"""zyx.utils.core"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Type,
    Tuple,
    TypeVar,
)

from pydantic import ValidationError
from pydantic_ai import (
    Agent,
    messages as _pydantic_ai_messages,
    models as _pydantic_ai_models,
    tools as _pydantic_ai_tools,
    toolsets as _pydantic_ai_toolsets,
)

from ..exceptions import (
    InvalidModelError,
)
from ..types import ModelParam, ToolParam, InstructionsParam
from ..processing.messages import (
    PydanticAIMessageAdapter,
    parse_string_to_messages,
    parse_openai_like_to_message,
    build_system_prompt,
)


DepsType = TypeVar("DepsType")


@lru_cache(maxsize=32)
def _create_agent_from_model(
    model: str | _pydantic_ai_models.Model,
    model_settings: _pydantic_ai_models.ModelSettings | None = None,
    deps_type: Type[DepsType] | None = None,
) -> Agent:
    """
    Creates a new Pydantic AI agent from a model name or model object.
    """
    return Agent(
        model=model,
        model_settings=model_settings,
        deps_type=deps_type,
    )  # type: ignore


def prepare_agent(
    model: ModelParam | None = None,
    model_settings: _pydantic_ai_models.ModelSettings | None = None,
    deps_type: Type[DepsType] | None = None,
) -> Agent:
    """
    Prepares a Pydantic AI agent from a model name or model object.
    """
    if not model:
        raise InvalidModelError(model)

    if isinstance(model, Agent):
        agent = model

        if model_settings:
            agent.model_settings = model_settings
        if deps_type:
            agent._deps_type = deps_type  # type: ignore

        return agent
    elif isinstance(model, (str, _pydantic_ai_models.Model)):
        return _create_agent_from_model(
            model=model,
            model_settings=model_settings,
            deps_type=deps_type,
        )

    else:
        raise InvalidModelError(model)


def prepare_message_history(
    context: Any,
) -> List[_pydantic_ai_messages.ModelMessage]:
    """
    Prepares a list of `pydantic_ai.ModelMessage` objects from a given
    context.

    This parses:
    - **str** (String content, that can contain role tags : [s]/[system], [u]/[user], [a]/[assistant])
    - **Dict[str, Any]** / **BaseModel** (A dictionary/pydantic model in the PydanticAI ModelMessage or OpenAI Chat Completions format)
    - **ModelMessage** (A PydanticAI ModelMessage object)

    into a list of `pydantic_ai.ModelMessage` objects.
    """
    if not isinstance(context, list):
        context = [context]

    parsed: List[_pydantic_ai_messages.ModelMessage] = []

    for item in context:
        if isinstance(item, str):
            parsed.extend(parse_string_to_messages(item))

        elif isinstance(item, Dict[str, Any]):
            if "parts" in item:
                try:
                    parsed.append(PydanticAIMessageAdapter.validate_python(item))
                except ValidationError as e:
                    raise ValueError(
                        "Recieved invalid dictionary representation of a message. ",
                        f"Error: {e}",
                    )

            else:
                parsed.append(parse_openai_like_to_message(item))

        elif isinstance(
            item,
            (
                _pydantic_ai_messages.ModelRequest,
                _pydantic_ai_messages.ModelResponse,
            ),
        ):
            parsed.append(item)

        else:
            raise ValueError(
                "Recieved invalid context item. ",
                f"Expected a string, dictionary, or ModelMessage, received: {type(item)}",
            )

    return parsed


def _split_instructions(
    instructions: InstructionsParam[DepsType] | None = None,
) -> Tuple[List[str], List[Callable]]:
    """
    Splits a list of instructions into a list of strings.
    """
    if not isinstance(instructions, list):
        instructions = [instructions]  # type: ignore

    instructions_list: List[str] = []
    functions_list: List[Callable] = []

    for i in instructions:  # type: ignore
        if isinstance(i, str):
            instructions_list.append(i)
        elif isinstance(i, (List[str])):
            instructions_list.extend(i)  # type: ignore
        elif inspect.isfunction(i):
            functions_list.append(i)

        else:
            raise ValueError(
                "Recieved invalid instruction type. ",
                f"Expected a string, list of strings, or callable, received: {type(i)}",
            )

    return instructions_list, functions_list


def prepare_context(
    context: Any,
    instructions: InstructionsParam[DepsType] | None = None,
    prepend_instructions: str | None = None,
    append_instructions: str | None = None,
    constraints_instructions: str | None = None,
) -> Dict[str, Any]:
    """
    Prepares a list of `pydantic_ai.ModelMessage` objects from a given
    context.

    Returns a dictionary with the following keys:
    - message_history : List[_pydantic_ai_messages.ModelMessage]
    - instructions : List[Callable]
    """
    messages: List[_pydantic_ai_messages.ModelMessage] = prepare_message_history(
        context
    )

    instructions_list: List[str] = []
    functions_list: List[Callable] = []

    if instructions:
        instructions_list, functions_list = _split_instructions(instructions)

    rendered_instructions = build_system_prompt(
        instructions=instructions_list,
        prepend_instructions=prepend_instructions,
        append_instructions=append_instructions,
        constraints=constraints_instructions,
    )

    if rendered_instructions:
        messages.insert(
            0, _pydantic_ai_messages.ModelRequest(parts=[rendered_instructions])
        )

    return {
        "message_history": messages,
        "instructions": functions_list,
    }


def prepare_tools(
    tools: List[ToolParam] | ToolParam | None = None,
) -> Dict[str, Any | None]:
    """
    Prepares an accepted tool type or list of tools into a dictionary
    with the following keys:

    - toolsets : List[AbstractToolset]
    - builtin_tools : List[AbstractBuiltinTool]
    """
    pydantic_ai_tools: List[_pydantic_ai_tools.Tool] = []
    toolsets: List[_pydantic_ai_toolsets.AbstractToolset] = []
    builtin_tools: List[_pydantic_ai_tools.AbstractBuiltinTool] = []

    if not tools:
        return {
            "toolsets": None,
            "builtin_tools": None,
        }

    if not isinstance(tools, list):
        tools = [tools]

    for tool in tools:
        if inspect.isfunction(tool):
            pydantic_ai_tools.append(_pydantic_ai_tools.Tool(function=tool))
        elif isinstance(tool, _pydantic_ai_tools.Tool):
            pydantic_ai_tools.append(tool)
        elif isinstance(tool, _pydantic_ai_toolsets.AbstractToolset):
            toolsets.append(tool)
        elif isinstance(tool, _pydantic_ai_tools.AbstractBuiltinTool):
            builtin_tools.append(tool)
        else:
            raise ValueError(
                "Recieved invalid tool type. ",
                f"Expected a function, Tool, AbstractToolset, or AbstractBuiltinTool, received: {type(tool)}",
            )

    if pydantic_ai_tools:
        toolsets.append(
            _pydantic_ai_toolsets.FunctionToolset(tools=pydantic_ai_tools)
        )

    return {
        "toolsets": toolsets if toolsets else None,
        "builtin_tools": builtin_tools if builtin_tools else None,
    }
