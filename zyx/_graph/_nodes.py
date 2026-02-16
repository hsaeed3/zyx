"""zyx._graph._nodes"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from typing import Any, List, TypeVar, Callable, Union, Generic, cast

from pydantic_graph.beta.step import StepContext
from pydantic_graph.nodes import End, GraphRunContext, BaseNode
from pydantic_ai.exceptions import UserError

from .._aliases import (
    PydanticAIAgentResult,
    PydanticAIAgentStream,
)
from ._ctx import (
    SemanticGraphDeps,
    SemanticGraphState,
    StreamFieldMapping,
)
from ._requests import SemanticGraphRequestTemplate

__all__ = (
    "AbstractSemanticNode",
    "make_generate_step",
    "make_stream_step",
    "run_v1_node_chain",
)


# temporary workaround used if a model provider doesnt support native structured output
_NATIVE_OUTPUT_UNSUPPORTED_ERROR = "Native structured output is not supported"


Deps = TypeVar("Deps")
Output = TypeVar("Output")


class AbstractSemanticNode(
    Generic[Deps, Output],
    BaseNode[
        SemanticGraphState[Output], SemanticGraphDeps[Deps, Output], End | Any
    ],
):
    """Base class for v1-style semantic nodes used in edit workflows."""

    async def execute_run(
        self,
        ctx: GraphRunContext[
            SemanticGraphState[Output], SemanticGraphDeps[Deps, Output]
        ],
        *,
        request: SemanticGraphRequestTemplate[Output],
        update_output: bool = True,
        output_fields: str | List[str] | None = None,
    ) -> PydanticAIAgentResult[Output | Any]:
        result = await _execute_with_fallback(
            ctx,
            request=request,
            streaming=False,
        )

        if update_output:
            ctx.state.output.update_from_pydantic_ai_result(
                result=cast(PydanticAIAgentResult[Any], result),
                fields=output_fields,
            )

        ctx.state.agent_runs.append(result)  # type: ignore[arg-type]
        ctx.state.usage.incr(result.usage())  # type: ignore[arg-type]
        return result  # type: ignore[return-value]

    async def execute_stream(
        self,
        ctx: GraphRunContext[
            SemanticGraphState[Output], SemanticGraphDeps[Deps, Output]
        ],
        *,
        request: SemanticGraphRequestTemplate[Output],
    ) -> AbstractAsyncContextManager[PydanticAIAgentStream[Any, Output | Any]]:
        stream_ctx = await _execute_with_fallback(
            ctx,
            request=request,
            streaming=True,
        )
        assert isinstance(stream_ctx, AbstractAsyncContextManager)
        return stream_ctx  # type: ignore[return-value]


async def _execute_run(
    ctx: StepContext[
        SemanticGraphState[Output], SemanticGraphDeps[Deps, Output], Any
    ],
    *,
    request: SemanticGraphRequestTemplate[Output],
    update_output: bool = True,
    output_fields: str | List[str] | None = None,
) -> PydanticAIAgentResult[Output | Any]:
    result = await _execute_with_fallback(
        ctx,
        request=request,
        streaming=False,
    )

    if update_output:
        ctx.state.output.update_from_pydantic_ai_result(
            result=cast(PydanticAIAgentResult[Any], result),
            fields=output_fields,
        )

    ctx.state.agent_runs.append(result)  # type: ignore[arg-type]
    ctx.state.usage.incr(result.usage())  # type: ignore[arg-type]
    return result  # type: ignore[return-value]


async def _execute_with_fallback(
    ctx: Any,
    *,
    request: SemanticGraphRequestTemplate[Output],
    streaming: bool,
) -> Union[
    PydanticAIAgentResult[Output | Any],
    AbstractAsyncContextManager[PydanticAIAgentStream[Any, Output | Any]],
]:
    params = request.render(ctx)
    observe = getattr(ctx.deps, "observe", None)
    handler = (
        getattr(observe, "event_stream_handler", None) if observe else None
    )

    try:
        if observe and hasattr(observe, "on_model_request"):
            observe.on_model_request()
        return await _do_run(
            ctx,
            params=params,
            streaming=streaming,
            handler=handler,
            observe=observe,
        )
    except UserError as e:
        if _NATIVE_OUTPUT_UNSUPPORTED_ERROR not in str(e):
            raise
        fallback_request = replace(request, native_output=False)
        fallback_params = fallback_request.render(ctx)
        return await _do_run(
            ctx,
            params=fallback_params,
            streaming=streaming,
            handler=handler,
            observe=observe,
        )


async def _do_run(
    ctx: Any,
    *,
    params: dict[str, Any],
    streaming: bool,
    handler: Any,
    observe: Any,
) -> Union[
    PydanticAIAgentResult[Output | Any],
    AbstractAsyncContextManager[PydanticAIAgentStream[Any, Output | Any]],
]:
    if streaming:
        return ctx.deps.agent.run_stream(
            **params, event_stream_handler=handler
        )
    if observe:
        async with ctx.deps.agent.iter(**params) as agent_run:
            async for node in agent_run:
                observe.on_node(node, agent_run.ctx)
        result = agent_run.result
        if result is None:
            raise ValueError("Agent run finished without a result.")
        return result
    return await ctx.deps.agent.run(**params)


def make_generate_step(
    request: SemanticGraphRequestTemplate[Output],
    *,
    update_output: bool = True,
    output_fields: str | List[str] | None = None,
) -> Callable[
    [
        StepContext[
            SemanticGraphState[Output], SemanticGraphDeps[Deps, Output], Any
        ]
    ],
    Any,
]:
    async def _step(
        ctx: StepContext[
            SemanticGraphState[Output], SemanticGraphDeps[Deps, Output], Any
        ],
    ) -> Any:
        result = await _execute_run(
            ctx,
            request=request,
            update_output=update_output,
            output_fields=output_fields,
        )
        return result.output

    return _step


def make_stream_step(
    request: SemanticGraphRequestTemplate[Output],
    *,
    update_output: bool = True,
    output_fields: str | List[str] | None = None,
) -> Callable[
    [
        StepContext[
            SemanticGraphState[Output], SemanticGraphDeps[Deps, Output], Any
        ]
    ],
    Any,
]:
    async def _step(
        ctx: StepContext[
            SemanticGraphState[Output], SemanticGraphDeps[Deps, Output], Any
        ],
    ) -> Any:
        stream_ctx = await _execute_with_fallback(
            ctx,
            request=request,
            streaming=True,
        )
        assert isinstance(stream_ctx, AbstractAsyncContextManager)

        stream = await stream_ctx.__aenter__()
        ctx.state.streams.append(stream)  # type: ignore
        ctx.state.stream_contexts.append(stream_ctx)

        if output_fields:
            fields = (
                output_fields
                if isinstance(output_fields, list)
                else [output_fields]
            )
            ctx.state.stream_field_mappings.append(
                StreamFieldMapping(
                    stream_index=len(ctx.state.streams) - 1,
                    fields=fields,
                    update_output=update_output,
                )
            )
        else:
            ctx.state.stream_field_mappings.append(
                StreamFieldMapping(
                    stream_index=len(ctx.state.streams) - 1,
                    fields=None,
                    update_output=update_output,
                )
            )

        return None

    return _step


async def run_v1_node_chain(
    start_node: BaseNode[Any, Any, Any],
    ctx: StepContext[
        SemanticGraphState[Output], SemanticGraphDeps[Deps, Output], Any
    ],
) -> Any:
    node: BaseNode[Any, Any, Any] = start_node
    while True:
        result = await node.run(
            GraphRunContext(state=ctx.state, deps=ctx.deps)
        )
        if isinstance(result, End):
            return result.data
        if not isinstance(result, BaseNode):
            raise ValueError(f"Invalid node transition: {type(result)}")
        node = result
