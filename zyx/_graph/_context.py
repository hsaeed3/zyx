"""zyx._graph._context"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Dict, List, Generic, Self, Type, TypeVar

from pydantic_graph import GraphRunContext

from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.output import NativeOutput
from pydantic_ai.settings import merge_model_settings

from .._processing._messages import (
    parse_context_to_pydantic_ai_messages,
    parse_instructions_as_system_prompt_parts,
)
from .._aliases import (
    PydanticAIAgent,
    PydanticAIAgentResult,
    PydanticAIAgentStream,
    PydanticAIModelSettings,
    PydanticAIModel,
    PydanticAIMessage,
    PydanticAITool,
    PydanticAIBuiltinTool,
    PydanticAIToolset,
    PydanticAIUsage,
    PydanticAIUsageLimits,
    PydanticAISystemPromptPart,
)
from .._utils._outputs import OutputBuilder
from .._processing._outputs import normalize_output_target
from ..context import Context
from ..resources.abstract import AbstractResource
from ..snippets import Snippet
from ..targets import Target


_logger = logging.getLogger(__name__)


Deps = TypeVar("Deps")
Output = TypeVar("Output")


@dataclass
class SemanticGraphDeps(Generic[Deps, Output]):
    """
    Immutable configuration / dependencies provided to a semantic operation's
    graph for execution.

    The primary purpose of this class is to provide a normalized representation
    of various user-provided parameters in the `pydantic_ai` format, as well as
    contain framework specific parameters.
    """

    agent: PydanticAIAgent
    """The `pydantic_ai.Agent` instance that will be invoked when executing the nodes
    within a semantic operation's graph."""

    target: Type[Output] | Output | Target[Output]
    """The target type or value that is treated as the final output of a semantic operation."""

    source: Any | Type[Any] | None = None
    """A source object that acts as the ground truth or primary context reference for
    specific semantic operations such as `parse` or `query`."""

    confidence: bool = False
    """Whether to include confidence scores in the `Result` of a semantic operation."""

    message_history: List[PydanticAIMessage] = field(default_factory=list)
    """Parsed message history from the `context` and `instructions` parameters of a
    semantic operation, in the `pydantic_ai.ModelMessage` format."""

    instructions: List[PydanticAISystemPromptPart] = field(
        default_factory=list
    )
    """User-provided instructions that are forwarded to the agent, these are parsed into
    `pydantic_ai.SystemPromptPart` objects when being passed to an agent."""

    toolsets: List[PydanticAIToolset] = field(default_factory=list)
    """Unified representation of all user-provided functions, `pydantic_ai` tools, builtin tools
    and toolsets.

    NOTE: All functions, tools and builtin tools are converted into a single `FunctionToolset`
    when passed to an agent."""

    deps: Deps | None = None
    """Reference to `deps` within `pydantic_ai` which can be passed to tools, messages and instructions
    through `pydantic_ai.RunContext`."""

    usage_limits: PydanticAIUsageLimits | None = None
    """The usage limits to set for a single agent run within the execution of a
    semantic operation's graph."""

    attachments: List[Snippet | AbstractResource] | None = None
    """A single or list of `Snippet` or `AbstractResource` objects that are provided to the agent.

    An attachment is a piece of content that is provided to the agent in a 'persistent' fashion,
    where it is templated/placed specifically to avoid context rot or loss. Furthermore, attachments that
    are `Resources` provide the agent with an ability to interact with/modify them, like artifacts.
    """

    _context_refs: Context | List[Context] | None = None
    """Reference to any `Context` objects that were found within the `context` parameter
    of a semantic operation."""

    @classmethod
    def prepare(
        cls,
        model: str | PydanticAIModel | PydanticAIAgent,
        model_settings: PydanticAIModelSettings | None = None,
        context: Any | List[Any] | None = None,
        source: Any | Type[Any] | None = None,
        target: Type[Output] | Output | None = None,
        confidence: bool = False,
        attachments: Snippet
        | AbstractResource
        | List[Snippet | AbstractResource]
        | None = None,
        instructions: str | Callable | List[str | Callable] | None = None,
        tools: Any | List[Any] | None = None,
        deps: Deps | None = None,
        usage_limits: PydanticAIUsageLimits | None = None,
    ) -> Self:
        """
        Prepares a new instance of `SemanticGraphDeps` through a standardized set of
        parameters shared by semantic operations.
        """
        agent: PydanticAIAgent | None = None

        if isinstance(model, PydanticAIAgent):
            agent = model
            if model_settings is not None:
                agent.model_settings = merge_model_settings(
                    base=agent.model_settings,
                    overrides=model_settings,
                )

        elif isinstance(model, (PydanticAIModel, str)):
            output_type = None
            if isinstance(target, Target):
                normalized = normalize_output_target(target.target)
                output_type = NativeOutput(
                    outputs=normalized,
                    name=target.name,
                    description=target.description,
                )

            agent = PydanticAIAgent(
                model=model,
                model_settings=model_settings,
                deps_type=type(deps) if deps is not None else None,
                **(
                    {"output_type": output_type}
                    if output_type is not None
                    else {}
                ),
            )  # type: ignore

        if not agent:
            raise ValueError(
                "Invalid 'runner' (model) provided when preparing semantic operation graph dependencies. Accepted formats for a model are:\n"
                "1. A string in the `pydantic_ai` model format (e.g. 'openai:gpt-4o-mini', etc.)\n"
                "2. A `pydantic_ai.models.Model` object\n"
                "3. A `pydantic_ai.Agent` object\n"
            )

        if confidence is True:
            agent = _ensure_confidence_supported_agent(agent)

        _context_refs: list[Context] = []

        if context:
            ctx_list = context if isinstance(context, list) else [context]
            _context_refs = [
                item for item in ctx_list if isinstance(item, Context)
            ]
            # Optionally infer deps from the most recent Context, if not yet set
            if _context_refs:
                last_ctx = _context_refs[-1]
                if (
                    getattr(last_ctx, "deps", None) is not None
                    and deps is None
                ):
                    deps = last_ctx.deps

        instructions_list = []
        if instructions is not None:
            # Guarantee list hygiene
            if not isinstance(instructions, list):
                instructions_list = [instructions]
            else:
                instructions_list = instructions

        rendered_instructions = parse_instructions_as_system_prompt_parts(
            instructions=instructions_list,
            deps=deps,
        )
        for ctx in _context_refs:
            rendered_instructions.extend(ctx.render_instructions())

        message_history = parse_context_to_pydantic_ai_messages(context)

        # Also aggregate toolsets from ALL Context objects if present
        ctx_toolsets = []
        for ctx in _context_refs:
            ctx_toolsets.extend(ctx.render_toolsets())

        function_tools = []
        toolsets = ctx_toolsets
        attachments_list: List[Any] = []

        if tools:
            if not isinstance(tools, list):
                tools = [tools]

            for tool in tools:
                if inspect.isfunction(tool):
                    function_tools.append(PydanticAITool(function=tool))
                elif isinstance(tool, (PydanticAIBuiltinTool, PydanticAITool)):
                    function_tools.append(tool)
                elif isinstance(tool, PydanticAIToolset):
                    toolsets.append(tool)
                elif isinstance(tool, AbstractResource):
                    toolsets.append(tool.get_toolset())
                else:
                    raise ValueError(
                        f"Invalid tool: {tool}. Accepted formats for tools are:\n"
                        "1. A function\n"
                        "2. A `pydantic_ai.tools.Tool` object\n"
                        "3. A `pydantic_ai.toolsets.AbstractToolset` object\n"
                    )

        if function_tools:
            ctx_toolsets.append(FunctionToolset(tools=function_tools))

        if attachments:
            if not isinstance(attachments, list):
                attachments = [attachments]
            for attachment in attachments:
                if isinstance(attachment, AbstractResource):
                    attachments_list.append(attachment)
                    toolsets.append(attachment.get_toolset())
                elif isinstance(attachment, Snippet):
                    attachments_list.append(attachment)
                else:
                    raise ValueError(
                        f"Invalid attachment: {attachment}. Accepted formats for attachments are:\n"
                        "1. A `Snippet`\n"
                        "2. A `Resource`"
                    )

        return cls(
            agent=agent,
            target=target,
            source=source,
            confidence=confidence,
            message_history=message_history,
            instructions=rendered_instructions,
            toolsets=toolsets,
            deps=deps,
            usage_limits=usage_limits,
            attachments=attachments_list,
            _context_refs=_context_refs,
        )


@dataclass
class SemanticGraphState(Generic[Output]):
    """
    Mutable object that is updated / changed on a node-to-node basis as a semantic operation
    is executed.
    """

    output: OutputBuilder[Output]
    """Builder object that tracks the state of the output `target`'s composition
    over the execution of a semantic operation's graph."""

    agent_runs: List[PydanticAIAgentResult] = field(default_factory=list)
    """Accumulation of all `pydantic_ai.AgentRunResult` objects that have been executed by nodes within
    the execution of a semantic operation's graph."""

    usage: PydanticAIUsage = field(default_factory=PydanticAIUsage)
    """Accumulated token usage across all agent runs within the execution of a semantic
    operation."""

    streams: List[PydanticAIAgentStream] = field(default_factory=list)
    """Streamed Agent runs that havent been consumed yet by the `Stream` result wrapper."""

    stream_contexts: List[Any] = field(default_factory=list)
    """Context managers that handle the consumption of the content within streams, used for
    cleanup."""

    stream_field_mappings: List[Dict[str, Any]] = field(default_factory=list)
    """Metadata about which fields each stream should update.

    Each entry contains:
    - stream_index: Index into self.streams
    - fields: List of field names to update, or None for all fields
    - update_output: Whether to update the output builder
    """

    @classmethod
    def prepare(cls, deps: SemanticGraphDeps[Deps, Output]) -> Self:
        """Prepares a new instance of `SemanticGraphState` using a provided
        `SemanticGraphDeps` instance."""
        if isinstance(deps.target, Target):
            return cls(
                output=OutputBuilder(target=deps.target.target),
            )
        return cls(
            output=OutputBuilder(target=deps.target),
        )


SemanticGraphContext = GraphRunContext[
    SemanticGraphState[Output], SemanticGraphDeps[Deps, Output]
]
"""RunContext type passed through the nodes of a semantic operation's graph."""


def _ensure_confidence_supported_agent(
    agent: PydanticAIAgent,
) -> PydanticAIAgent:
    """
    Ensures that a given `pydantic_ai.Agent` is configured to return confidence
    scores, when it is active.
    """
    model = agent._get_model(agent.model)
    model_name = model.model_name

    if not model.client or model.client.__class__.__name__ not in (  # type: ignore[attr-defined]
        "OpenAI",
        "AsyncOpenAI",
    ):
        _logger.warning(
            f"Confidence scoring is only supported with OpenAI or OpenAI-like models. "
            f"Model '{model}' may not return log-probabilities, so confidence will be inaccurate."
        )
        return agent
    else:
        if model.__class__.__name__ == "OpenAIChatModel":
            from pydantic_ai.models.openai import (
                OpenAIResponsesModel,
                OpenAIResponsesModelSettings,
            )
            from pydantic_ai.providers.openai import OpenAIProvider

            _logger.debug(
                f"Attempting to switch model class for {model_name} to use OpenAIResponsesModel."
            )
            agent._model = OpenAIResponsesModel(
                model_name=model_name,
                provider=OpenAIProvider(
                    openai_client=model.client  # type: ignore[attr-defined]
                ),
            )
            agent.model_settings = merge_model_settings(
                base=agent.model_settings,
                overrides=OpenAIResponsesModelSettings(
                    openai_logprobs=True, openai_top_logprobs=10
                ),
            )
            return agent

    return agent
