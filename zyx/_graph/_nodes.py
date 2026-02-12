"""zyx._graph._nodes"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, List, Generic, Self, TypeVar

from pydantic_graph import (
    BaseNode,
    End,
)

from .._aliases import (
    PydanticAIAgentResult,
    PydanticAIAgentStream,
    PydanticAIUsage,
)
from ._context import (
    SemanticGraphDeps,
    SemanticGraphState,
    SemanticGraphContext,
)
from ._requests import SemanticGraphRequestTemplate


Deps = TypeVar("Deps")
Output = TypeVar("Output")


@dataclass
class AbstractSemanticNode(
    BaseNode[
        SemanticGraphState[Output],
        SemanticGraphDeps[Deps, Output],
        End[Output] | Any,
    ],
    ABC,
    Generic[Deps, Output],
):
    """
    Abstract base class for all nodes that make up the execution graph
    of a semantic operation.
    """

    @abstractmethod
    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> Self | End[Output]:
        """
        Execute this node and return the next node or the final output
        (End) of the graph.
        """

    async def execute_run(
        self,
        ctx: SemanticGraphContext[Output, Deps],
        *,
        request: SemanticGraphRequestTemplate[Output],
        update_output: bool = True,
        output_fields: str | List[str] | None = None,
    ) -> PydanticAIAgentResult[Output | Any]:
        """
        Executes a single agent run and updates the state of the graph and
        output based on if specified, and according to fields generated in this
        request.
        """
        params = request.render(ctx)

        try:
            result = await ctx.deps.agent.run(**params)
        except Exception as e:
            raise e

        if update_output:
            ctx.state.output.update_from_pydantic_ai_result(
                result=result, fields=output_fields
            )

        ctx.state.agent_runs.append(result)
        ctx.state.usage.incr(result.usage())
        return result

    def execute_stream(
        self,
        ctx: SemanticGraphContext[Output, Deps],
        *,
        request: SemanticGraphRequestTemplate[Output],
    ) -> AbstractAsyncContextManager[PydanticAIAgentStream[Any, Output | Any]]:
        """
        Creates an async context manager for a streaming agent run.

        Returns the context manager from `agent.run_stream()` which must be
        entered with `async with` or `__aenter__()` to get the actual
        `StreamedRunResult`.
        """
        params = request.render(ctx)
        return ctx.deps.agent.run_stream(**params)


@dataclass
class SemanticGenerateNode(AbstractSemanticNode[Deps, Output]):
    """
    'Run' or generation node that executes a single agent run to return
    either a full or partial representation of the `target` output, or
    some other AgentRunResult.
    """

    request: SemanticGraphRequestTemplate[Output]
    """Specific request context specific to this node."""

    update_output: bool = True
    """Whether to update the output of the graph based on the result of the run."""

    output_fields: str | List[str] | None = None
    """If specified, only update the output of the graph based on the specified fields."""

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output]:
        """
        Execute the node and return the next node or the final output
        (End) of the graph.
        """
        result = await self.execute_run(
            ctx=ctx,
            request=self.request,
            update_output=self.update_output,
            output_fields=self.output_fields,
        )

        return End(result.output)


@dataclass
class SemanticStreamNode(AbstractSemanticNode[Deps, Output]):
    """
    Run node that executes a single agent run to return a StreamedRunResult,
    that represents either a full or partial representation of the `target`
    output, or some other result.

    Unlike SemanticGenerateNode, this node DOES NOT consume the stream.
    Instead, it stores the stream in graph state for the Stream wrapper
    to consume, enabling real-time streaming at the graph level.
    """

    request: SemanticGraphRequestTemplate[Output]
    """Specific request context specific to this node."""

    update_output: bool = True
    """Whether to update the output of the graph based on the result of the stream."""

    output_fields: str | List[str] | None = None
    """If specified, only update the output of the graph based on the specified fields."""

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output]:
        """
        Execute the node and create a stream for consumption.

        This method:
        1. Executes execute_stream() to get a StreamedRunResult
        2. Stores the stream in ctx.state.streams for later consumption
        3. Stores metadata about which fields this stream should update
        4. Returns End (graph continues to next node or finishes)

        The actual stream consumption happens in the Stream wrapper,
        allowing real-time streaming across multiple nodes.
        """
        # Get the stream context manager from agent.run_stream()
        stream_ctx = self.execute_stream(ctx=ctx, request=self.request)

        # Enter the context manager to get the actual StreamedRunResult
        stream = await stream_ctx.__aenter__()

        # Store stream info for the Stream wrapper to consume
        ctx.state.streams.append(stream)  # type: ignore
        ctx.state.stream_contexts.append(stream_ctx)  # For cleanup

        # Store metadata about this stream
        if self.output_fields:
            ctx.state.stream_field_mappings.append(
                {
                    "stream_index": len(ctx.state.streams) - 1,
                    "fields": self.output_fields
                    if isinstance(self.output_fields, list)
                    else [self.output_fields],
                    "update_output": self.update_output,
                }
            )
        else:
            ctx.state.stream_field_mappings.append(
                {
                    "stream_index": len(ctx.state.streams) - 1,
                    "fields": None,  # All fields
                    "update_output": self.update_output,
                }
            )

        return End(None)  # type: ignore
