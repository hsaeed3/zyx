"""zyx.operations.make"""

from __future__ import annotations

import random
from typing import List, Literal, TypeVar, overload

from .._aliases import (
    PydanticAIInstructions,
    PydanticAIModelSettings,
    PydanticAIUsageLimits,
)
from .._graph import (
    SemanticGraph,
    SemanticGraphDeps,
    SemanticGraphState,
    SemanticGraphRequestTemplate,
    SemanticGenerateNode,
    SemanticStreamNode,
)
from ..result import Result
from ..stream import Stream
from .._types import (
    ModelParam,
    TargetParam,
    ContextType,
    ToolType,
    AttachmentType,
)


Deps = TypeVar("Deps")
Output = TypeVar("Output")


_DISTILLATION_KEYWORDS: list[str] = [
    "concise",
    "succinct",
    "evocative",
    "original",
    "minimal",
    "minimalist",
    "uncommon",
    "unusual",
    "abstract",
    "abstractive",
    "poetic",
    "lyrical",
    "philosophical",
    "deep",
    "thoughtful",
    "profound",
    "inspiring",
    "motivational",
    "educational",
    "enlightening",
    "explanatory",
    "analytical",
    "critical",
    "evaluative",
    "judgemental",
    "biased",
    "subjective",
    "opinionated",
    "creative",
    "imaginative",
    "curious",
    "insightful",
    "distinctive",
    "descriptive",
    "innovative",
    "strange",
    "objective",
    "reflective",
]


def prepare_make_graph(
    deps: SemanticGraphDeps[Deps, Output],
    state: SemanticGraphState[Output],
    randomize: bool = False,
    stream: bool = False,
) -> SemanticGraph[Output]:
    """
    Prepares a `SemanticGraph` Graph for the `make` operation, depending on
    it's set configuration.
    """

    randomization_system_prompt = (
        f"Ensure your final response is {random.choice(_DISTILLATION_KEYWORDS)} and {random.choice(_DISTILLATION_KEYWORDS)}."
        "Although you are free to be creative, try to be as realistic as possible to the given response schema."
        "Your goal is uncommon responses that are believable. Never use the first 2-3 ideas that come to mind."
    )

    if state.output.normalized is str:
        if not deps.message_history and not deps.instructions:
            raise ValueError(
                "Using `make()` with `target=str` requires providing either `context` or `instructions`. Please provide"
                "one of these parameters."
            )

    if not deps.message_history and not deps.instructions:
        request = SemanticGraphRequestTemplate(
            system_prompt_additions=randomization_system_prompt,
            native_output=False if deps.toolsets else True,
        )
        if not deps.agent.model_settings:
            deps.agent.model_settings = {"temperature": 0.7}
    else:
        request = SemanticGraphRequestTemplate(
            system_prompt_additions=randomization_system_prompt
            if randomize
            else None,
            native_output=False if deps.toolsets else True,
        )
        if randomize and not deps.agent.model_settings:
            deps.agent.model_settings = {"temperature": 0.7}

    if stream:
        nodes = [SemanticStreamNode]
        start_node = SemanticStreamNode(request=request)
    else:
        nodes = [SemanticGenerateNode]
        start_node = SemanticGenerateNode(request=request)
    return SemanticGraph(nodes=nodes, start=start_node, state=state, deps=deps)


@overload
async def amake(
    target: TargetParam[Output] = ...,
    context: ContextType | List[ContextType] | None = ...,
    *,
    randomize: bool = ...,
    confidence: bool = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    attachments: AttachmentType | List[AttachmentType] | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    stream: Literal[True],
) -> Stream[Output]: ...


@overload
async def amake(
    target: TargetParam[Output] = ...,
    context: ContextType | List[ContextType] | None = ...,
    *,
    randomize: bool = ...,
    confidence: bool = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    attachments: AttachmentType | List[AttachmentType] | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    stream: Literal[False] = False,
) -> Result[Output]: ...


async def amake(
    target: TargetParam[Output] = str,  # type: ignore[assignment]
    context: ContextType | List[ContextType] | None = None,
    *,
    randomize: bool = False,
    confidence: bool = False,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: PydanticAIModelSettings | None = None,
    attachments: AttachmentType | List[AttachmentType] | None = None,
    instructions: PydanticAIInstructions | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    usage_limits: PydanticAIUsageLimits | None = None,
    stream: bool = False,
) -> Result[Output] | Stream[Output]:
    """Asynchronously generate ('make') a value of the provided `target` type using
    a model or Pydantic AI agent.

    Args:
        target (TargetParam[Output]): The target type or value to generate. Defaults to str.
        context (ContextType | List[ContextType] | None): The context to use for the operation. Defaults to None.
        randomize (bool): Injects a simple randomization instruction for more diverse outputs. This is automatically
            added if no context or instructions are provided. Defaults to False.
        confidence (bool): Whether to include confidence scores in the result of the operation. This is currently only
            supported for OpenAI or OpenAI-like models. Defaults to False.
        model (ModelParam): The model to use for the operation. This can be a string, Pydantic AI model,
            or Pydantic AI agent. Defaults to "openai:gpt-4o-mini".
        model_settings (PydanticAIModelSettings | None): The model settings to use for the operation. Defaults to None.
        attachments (AttachmentType | List[AttachmentType] | None): A single or list of `Snippet` or `AbstractResource` objects that are provided to the agent.
            An attachment is a piece of content that is provided to the agent in a 'persistent' fashion,
            where it is templated/placed specifically to avoid context rot or loss. Furthermore, attachments that
            are `Resources` provide the agent with an ability to interact with/modify them, like artifacts. Defaults to None.
        instructions (PydanticAIInstructions | None): The instructions to use for the operation. Defaults to None.
        tools (ToolType | List[ToolType] | None): The tools to use for the operation. Defaults to None.
        deps (Deps | None): Reference to `deps` in `pydantic_ai.RunContext`, that can be passed to messages,
            tools and instructions. Defaults to None.
        usage_limits (PydanticAIUsageLimits | None): The usage limits to use for the operation. Defaults to None.
        stream (bool): Whether to stream the output of the operation. Defaults to False.

    Returns:
        Result[Output] | Stream[Output]: The result or stream of the operation.
    """
    graph_deps = SemanticGraphDeps.prepare(
        model=model,
        model_settings=model_settings,
        context=context,
        instructions=instructions,  # type: ignore[arg-type]
        tools=tools,
        confidence=confidence,
        target=target,
        deps=deps,
        usage_limits=usage_limits,
        attachments=attachments,
    )
    state = SemanticGraphState.prepare(deps=graph_deps)

    graph = prepare_make_graph(
        deps=graph_deps, state=state, randomize=randomize, stream=stream
    )

    if stream:
        return await graph.stream()
    else:
        return await graph.run()


@overload
def make(
    target: TargetParam[Output] = ...,
    context: ContextType | List[ContextType] | None = ...,
    *,
    randomize: bool = ...,
    confidence: bool = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    attachments: AttachmentType | List[AttachmentType] | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    stream: Literal[True],
) -> Stream[Output]: ...


@overload
def make(
    target: TargetParam[Output] = ...,
    context: ContextType | List[ContextType] | None = ...,
    *,
    randomize: bool = ...,
    confidence: bool = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    attachments: AttachmentType | List[AttachmentType] | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    stream: Literal[False] = False,
) -> Result[Output]: ...


def make(
    target: TargetParam[Output] = str,  # type: ignore[assignment]
    context: ContextType | List[ContextType] | None = None,
    *,
    randomize: bool = False,
    confidence: bool = False,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: PydanticAIModelSettings | None = None,
    attachments: AttachmentType | List[AttachmentType] | None = None,
    instructions: PydanticAIInstructions | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    usage_limits: PydanticAIUsageLimits | None = None,
    stream: bool = False,
) -> Result[Output] | Stream[Output]:
    """Generate ('make') a value of the provided `target` type using
    a model or Pydantic AI agent.

    Args:
        target (TargetParam[Output]): The target type or value to generate. Defaults to str.
        context (ContextType | List[ContextType] | None): The context to use for the operation. Defaults to None.
        randomize (bool): Injects a simple randomization instruction for more diverse outputs. This is automatically
            added if no context or instructions are provided. Defaults to False.
        confidence (bool): Whether to include confidence scores in the result of the operation. This is currently only
            supported for OpenAI or OpenAI-like models. Defaults to False.
        model (ModelParam): The model to use for the operation. This can be a string, Pydantic AI model,
            or Pydantic AI agent. Defaults to "openai:gpt-4o-mini".
        model_settings (PydanticAIModelSettings | None): The model settings to use for the operation. Defaults to None.
        attachments (AttachmentType | List[AttachmentType] | None): A single or list of `Snippet` or `AbstractResource` objects that are provided to the agent.
            An attachment is a piece of content that is provided to the agent in a 'persistent' fashion,
            where it is templated/placed specifically to avoid context rot or loss. Furthermore, attachments that
            are `Resources` provide the agent with an ability to interact with/modify them, like artifacts. Defaults to None.
        instructions (PydanticAIInstructions | None): The instructions to use for the operation. Defaults to None.
        tools (ToolType | List[ToolType] | None): The tools to use for the operation. Defaults to None.
        deps (Deps | None): Reference to `deps` in `pydantic_ai.RunContext`, that can be passed to messages,
            tools and instructions. Defaults to None.
        usage_limits (PydanticAIUsageLimits | None): The usage limits to use for the operation. Defaults to None.
        stream (bool): Whether to stream the output of the operation. Defaults to False.

    Returns:
        Result[Output] | Stream[Output]: The result or stream of the operation.
    """
    graph_deps = SemanticGraphDeps.prepare(
        model=model,
        model_settings=model_settings,
        context=context,
        instructions=instructions,  # type: ignore[arg-type]
        tools=tools,
        confidence=confidence,
        target=target,
        deps=deps,
        usage_limits=usage_limits,
        attachments=attachments,
    )
    state = SemanticGraphState.prepare(deps=graph_deps)

    graph = prepare_make_graph(
        deps=graph_deps, state=state, randomize=randomize, stream=stream
    )

    if stream:
        return graph.stream_sync()
    else:
        return graph.run_sync()
