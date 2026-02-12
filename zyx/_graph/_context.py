"""zyx._graph._context"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Generic, Type, TypeVar


from .._aliases import (
    PydanticAIAgent,
    PydanticAIAgentResult,
    PydanticAIAgentStream,
    PydanticAIInstructions,
    PydanticAIMessage,
    PydanticAIToolset,
    PydanticAIUsage,
    PydanticAIUsageLimits,
)


Deps = TypeVar("Deps")
Output = TypeVar("Output")


@dataclass
class SemanticGraphRequestTemplate(Generic[Output]):
    """
    Template class used by various steps within a semantic operation's graph, to build
    the context of a single request.
    """

    output_type: Type[Output] | None = None
    """An override to the `output_type` parameter within a single step/agent run within a
    graph."""

    system_prompt_additions: str | list[str] | None = None
    """Dynamic instructions that can be applied onto the existing system prompt/instructions
    within `SemanticGraphDeps`."""

    user_prompt_additions: str | list[str] | None = None
    """Dynamic user prompt content to append to the message history of a request."""

    user_prompt_template: str | None = None
    """A template string that is used **after** the result of a node is completed, when the
    node's context is added to the message history of an agent run. This allows complex multi
    step requests to be templated in a cleaner manner within the message history.

    This string should contain a single `{prompt}` variable placeholder."""

    agent_result_template: str | None = None
    """A template string that is used **after** the result of a node is generated, allowing
    framing of complex or messy agent outputs in a cleaner manner within the message history
    accumulated throughout a semantic operation's graph.

    This string should contain a single `{result}` variable placeholder, but can contain
    additional variables based on the operation's own requirements."""

    toolsets: List[PydanticAIToolset] | None = None
    """Any operation-specific toolsets that can be included within the request."""

    include_source_context: bool = True
    """Whether to include context about a given `source` within the system prompt of a
    request. (If one is provided)"""

    include_output_context: bool = False
    """Whether to provide context about the current state of the output within the system
    prompt of a request."""

    include_user_toolsets: bool = True
    """Whether user-provided tools/toolsets should be included within this request."""


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

    target: Type[Output] | Output
    """The target type or value that is treated as the final output of a semantic operation."""

    source: Any | Type[Any] | None = None
    """A source object that acts as the ground truth or primary context reference for
    specific semantic operations such as `parse` or `query`."""

    confidence: bool = False
    """Whether to include confidence scores in the `Result` of a semantic operation."""

    message_history: List[PydanticAIMessage] = field(default_factory=list)
    """Parsed message history from the `context` and `instructions` parameters of a
    semantic operation, in the `pydantic_ai.ModelMessage` format."""

    instructions: PydanticAIInstructions | None = None
    """User-provided instructions that are forwarded to the agent. These instructions are rendered
    into a single `SystemPromptPart` when being passed to an agent."""

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


@dataclass
class SemanticGraphState(Generic[Output]):
    """
    Mutable object that is updated / changed on a node-to-node basis as a semantic operation
    is executed.
    """

    template: SemanticGraphRequestTemplate[Output] | None = None
    """The operation-provided request template to use for the next node(s)."""

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
