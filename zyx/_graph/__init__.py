"""zyx._graph"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, Type, TypeVar

from pydantic_graph import End, Graph, GraphRun

from pydantic_ai import _output as _pydantic_ai_output
from pydantic_ai import _utils as _pydantic_ai_utils
from pydantic_ai.exceptions import ModelRetry

from ._context import (
    SemanticGraphDeps,
    SemanticGraphState,
    SemanticGraphContext,
)
from ._nodes import (
    AbstractSemanticNode,
    SemanticGenerateNode,
    SemanticStreamNode,
)
from ._requests import SemanticGraphRequestTemplate
from ..targets import Target
from ..result import Result
from ..stream import Stream

__all__ = (
    "SemanticGraphRun",
    "SemanticGraph",
    "AbstractSemanticNode",
    "SemanticGenerateNode",
    "SemanticStreamNode",
    "SemanticGraphRequestTemplate",
    "SemanticGraphDeps",
    "SemanticGraphState",
    "SemanticGraphContext",
)


Deps = TypeVar("Deps")
Output = TypeVar("Output")


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create event loop for sync operations."""
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop


@dataclass
class SemanticGraphRun(Generic[Output]):
    """
    Step-by-step execution of a semantic operation's graph. This wraps the `GraphRun`
    class from `pydantic_graph` to provide specific context and patterns for
    a semantic operation.
    """

    _graph_run: GraphRun[
        SemanticGraphState[Output],
        SemanticGraphDeps[Deps, Output],
        End[Output] | Any,
    ]

    @property
    def next_node(self) -> AbstractSemanticNode[Deps, Output] | End[Output]:
        """The next node to be executed in the graph."""
        return self._graph_run.next_node  # type: ignore

    @property
    def result(self) -> Output | None:
        """The final result if the run has ended, otherwise None."""
        if self._graph_run.result is not None:
            return self._graph_run.state.output.finalize()
        return None

    @property
    def state(self) -> SemanticGraphState[Output]:
        """The current state of the graph."""
        return self._graph_run.state

    @property
    def deps(self) -> SemanticGraphDeps[Deps, Output]:
        """The dependencies of the graph."""
        return self._graph_run.deps  # type: ignore

    def __aiter__(self):
        return self

    async def __anext__(
        self,
    ) -> AbstractSemanticNode[Deps, Output] | End[Output | Any]:
        """The next node to be executed in the graph."""
        node = await anext(self._graph_run)
        return node


class SemanticGraph(Generic[Output]):
    """
    Execution wrapper for a semantic operation graph.

    Wraps pydantic_graph's Graph with zyx's state/deps model and provides
    clean execution methods that return properly formatted results.
    """

    def __init__(
        self,
        *,
        nodes: Sequence[Type[AbstractSemanticNode[Deps, Output]]],
        start: AbstractSemanticNode[Deps, Output],
        state: SemanticGraphState[Output],
        deps: SemanticGraphDeps[Deps, Output],
    ) -> None:
        """Create a semantic operation graph.

        Args:
            nodes : The nodes that make up the graph.
            start : The start node of the graph.
            state : The initial state of the graph.
            deps : The dependencies of the graph.
        """
        self._nodes = nodes
        self._start = start
        self._state = state
        self._deps = deps
        self._graph: Graph | None = None

    def prepare_graph(self) -> Graph:
        if self._graph is None:
            self._graph = Graph(
                nodes=self._nodes,  # type: ignore
            )
        return self._graph

    def _auto_update_context(self, res: Result[Output]) -> None:
        """Write messages back to the originating ``Context`` if applicable."""
        ctx_ref = getattr(self._deps, "_context_ref", None)
        if (
            ctx_ref is not None
            and getattr(ctx_ref, "update", True)
            and res.raw
        ):
            ctx_ref.update_from_pydantic_ai_result(res.raw[-1])

    async def run(self) -> Result[Output]:
        """Execute the graph to completion and return the final result."""
        graph = self.prepare_graph()

        cleanup = _install_target_hooks(self._deps)

        try:
            if self._start and self._state and self._deps:
                result = await graph.run(
                    start_node=self._start,  # type: ignore
                    state=self._state,  # type: ignore
                    deps=self._deps,  # type: ignore
                )
        except Exception as e:
            _run_error_hooks(self._deps, e)
            raise
        finally:
            cleanup()

        res = Result(
            result.output,
            raw=result.state.agent_runs,  # type: ignore[attr-defined]
        )
        self._auto_update_context(res)
        return res  # type: ignore

    def run_sync(self) -> Result[Output]:
        """Execute the graph to completion and return the final result."""
        graph = self.prepare_graph()

        cleanup = _install_target_hooks(self._deps)

        try:
            if self._start and self._state and self._deps:
                result = graph.run_sync(
                    start_node=self._start,  # type: ignore
                    state=self._state,  # type: ignore
                    deps=self._deps,  # type: ignore
                )
        except Exception as e:
            _run_error_hooks(self._deps, e)
            raise
        finally:
            cleanup()

        res = Result(
            result.output,
            raw=result.state.agent_runs,  # type: ignore[attr-defined]
        )
        self._auto_update_context(res)
        return res  # type: ignore

    async def stream(self, *, exclude_none: bool = False) -> Stream[Output]:
        """Execute the graph with streaming nodes and return a Stream wrapper.

        The graph should contain `SemanticStreamNode` nodes that store their
        streams in `ctx.state.streams` during execution. After the graph runs
        to completion, the accumulated streams are collected and wrapped in a
        `Stream` object for consumption.

        Args:
            exclude_none: If ``True``, the resulting ``Stream`` will omit
                ``None``-valued fields when finalizing the output.

        Returns:
            A `Stream[Output]` that can be used to consume the streamed data
            via `stream_text()`, `stream_partial()`, etc.
        """
        graph = self.prepare_graph()

        cleanup = _install_target_hooks(self._deps)

        try:
            if self._start and self._state and self._deps:
                await graph.run(
                    start_node=self._start,  # type: ignore
                    state=self._state,  # type: ignore
                    deps=self._deps,  # type: ignore
                )
        except Exception as e:
            _run_error_hooks(self._deps, e)
            raise
        finally:
            cleanup()

        return Stream(
            _builder=self._state.output,
            _streams=self._state.streams,
            _field_mappings=self._state.stream_field_mappings,
            _stream_contexts=self._state.stream_contexts,
            _exclude_none=exclude_none,
        )

    def stream_sync(self, *, exclude_none: bool = False) -> Stream[Output]:
        """Synchronous wrapper around `stream()`.

        Returns a `Stream[Output]` whose sync iteration methods (`text()`,
        `partial()`, `field()`, `finish()`) can be used from non-async code.

        Args:
            exclude_none: Forwarded to ``stream()``.

        Raises:
            RuntimeError: If called from within an already-running async context.
        """
        loop = _get_event_loop()
        return loop.run_until_complete(self.stream(exclude_none=exclude_none))


def _install_target_hooks(
    deps: SemanticGraphDeps[Any, Any],
) -> Callable[[], None]:
    target = deps.target
    if not isinstance(target, Target):
        return lambda: None

    validators = _build_target_output_validators(target)
    if not validators:
        return lambda: None

    agent = deps.agent
    start = len(agent._output_validators)
    agent._output_validators.extend(validators)

    def _cleanup() -> None:
        del agent._output_validators[start:]

    return _cleanup


def _build_target_output_validators(
    target: Target[Any],
) -> list[_pydantic_ai_output.OutputValidator[Any, Any]]:
    field_hooks = target._field_hooks
    complete_hooks = target._prebuilt_hooks.get("complete", [])

    if not field_hooks and not complete_hooks:
        return []

    async def _validator(ctx, data):
        value = data

        # Field hooks
        if field_hooks:
            if hasattr(value, "model_copy"):
                updates = {}
                for field, hooks in field_hooks.items():
                    if field == "__self__":
                        value, did_update = await _apply_hooks(
                            hooks, ctx, value, field
                        )
                        if not did_update:
                            value = data
                        continue
                    if not hasattr(value, field):
                        continue
                    current = getattr(value, field)
                    if current is None:
                        continue
                    updated, did_update = await _apply_hooks(
                        hooks, ctx, current, field
                    )
                    if did_update:
                        updates[field] = updated
                if updates:
                    value = value.model_copy(update=updates)
            elif isinstance(value, dict):
                for field, hooks in field_hooks.items():
                    if field == "__self__":
                        value, did_update = await _apply_hooks(
                            hooks, ctx, value, field
                        )
                        if not did_update:
                            value = data
                        continue
                    if field not in value or value[field] is None:
                        continue
                    updated, did_update = await _apply_hooks(
                        hooks, ctx, value[field], field
                    )
                    if did_update:
                        value[field] = updated
            else:
                if "__self__" in field_hooks:
                    value, did_update = await _apply_hooks(
                        field_hooks["__self__"], ctx, value, "__self__"
                    )
                    if not did_update:
                        value = data

        # Complete hooks
        if complete_hooks:
            value, did_update = await _apply_hooks(
                complete_hooks, ctx, value, "__complete__"
            )
            if not did_update:
                value = data

        return value

    return [_pydantic_ai_output.OutputValidator(_validator)]


async def _apply_hooks(hooks, ctx, value, field_name: str):
    current = value
    did_update = False
    for fn, retry, update in hooks:
        try:
            takes_ctx = len(inspect.signature(fn).parameters) > 1
            if takes_ctx:
                args = (ctx, current)
            else:
                args = (current,)

            if _pydantic_ai_utils.is_async_callable(fn):
                next_value = await fn(*args)
            else:
                next_value = fn(*args)
            current = next_value
            if update:
                did_update = True
        except Exception as e:
            if retry:
                raise ModelRetry(
                    f"Target hook failed for '{field_name}': {e}"
                ) from e
            raise
    return current, did_update


def _run_error_hooks(
    deps: SemanticGraphDeps[Any, Any], error: BaseException
) -> None:
    target = deps.target
    if not isinstance(target, Target):
        return

    hooks = target._prebuilt_hooks.get("error", [])
    if not hooks:
        return

    for fn, _retry, _update in hooks:
        try:
            takes_ctx = len(inspect.signature(fn).parameters) > 1
            if takes_ctx:
                fn(deps)
            else:
                fn(deps)
        except Exception:
            pass
