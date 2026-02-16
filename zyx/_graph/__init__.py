"""zyx._graph"""

from __future__ import annotations

import asyncio
import threading
import inspect
from types import SimpleNamespace
from typing import Any, Callable, Generic, Sequence, TypeVar

from pydantic_graph.beta.graph import Graph
from pydantic_graph.beta.graph_builder import GraphBuilder
from pydantic_graph.nodes import End

from pydantic_ai import _output as _pydantic_ai_output
from pydantic_ai import _utils as _pydantic_ai_utils
from pydantic_ai.exceptions import ModelRetry

from ._ctx import (
    SemanticGraphDeps,
    SemanticGraphState,
    SemanticGraphContext,
    GraphHooks,
)
from ._nodes import (
    AbstractSemanticNode,
    make_generate_step,
    make_stream_step,
    run_v1_node_chain,
)
from ._requests import SemanticGraphRequestTemplate
from ..targets import Target
from ..result import Result
from ..stream import Stream

__all__ = (
    "SemanticGraph",
    "AbstractSemanticNode",
    "End",
    "make_generate_step",
    "make_stream_step",
    "run_v1_node_chain",
    "SemanticGraphRequestTemplate",
    "SemanticGraphDeps",
    "SemanticGraphState",
    "SemanticGraphContext",
    "GraphHooks",
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


def _run_coro_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - re-raised in caller
            error_box["error"] = exc

    thread = threading.Thread(target=_runner, name="zyx-asyncio-run")
    thread.start()
    thread.join()

    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


class SemanticGraph(Generic[Output]):
    """
    Execution wrapper for a semantic operation graph.

    Wraps pydantic_graph's Graph with zyx's state/deps model and provides
    clean execution methods that return properly formatted results.
    """

    def __init__(
        self,
        *,
        steps: Sequence[Callable[..., Any]],
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
        self._steps = steps
        self._state = state
        self._deps = deps
        self._graph: Graph | None = None

    def prepare_graph(self) -> Graph:
        if self._graph is None:
            builder = GraphBuilder(
                state_type=type(self._state),
                deps_type=type(self._deps),
                input_type=type(None),
                output_type=Any,
            )
            prev = builder.start_node
            for idx, step in enumerate(self._steps):
                step_node = builder.step(call=step, node_id=f"step_{idx}")
                builder.add(builder.edge_from(prev).to(step_node))
                prev = step_node
            builder.add(builder.edge_from(prev).to(builder.end_node))
            self._graph = builder.build()
        return self._graph

    def _auto_update_context(self, res: Result[Output]) -> None:
        """Write messages back to the originating ``Context`` if applicable."""
        ctx_refs = getattr(self._deps, "_context_refs", None) or []
        if not isinstance(ctx_refs, list):
            ctx_refs = [ctx_refs]
        context_additions = getattr(self._deps, "_context_additions", []) or []

        for ctx_ref in ctx_refs:
            if (
                ctx_ref is None
                or not getattr(ctx_ref, "update", True)
                or not res.raw
            ):
                continue

            if context_additions:
                ctx_ref.extend_messages(context_additions)

            semantic_renderer = getattr(self._deps, "semantic_renderer", None)
            if semantic_renderer:
                semantic = semantic_renderer(res, self._state, self._deps)
                if isinstance(semantic, str):
                    res._semantic_message = semantic
                    ctx_ref.add_assistant_message(semantic)
                    continue
                if isinstance(semantic, list):
                    ctx_ref.extend_messages(semantic)
                    continue
                if semantic is not None:
                    ctx_ref.extend_messages([semantic])
                    continue
            ctx_ref.update_from_pydantic_ai_result(res.raw[-1])

    async def run(self) -> Result[Output]:
        """Execute the graph to completion and return the final result."""
        graph = self.prepare_graph()

        cleanup = _install_target_hooks(self._deps)
        _run_graph_start_hooks(self._deps, self._state)

        try:
            if self._state and self._deps:
                await graph.run(
                    state=self._state,
                    deps=self._deps,
                    inputs=None,
                )
        except Exception as e:
            _run_graph_error_hooks(self._deps, e, self._state)
            _run_error_hooks(self._deps, e)
            raise
        finally:
            cleanup()

        res = Result(
            self._state.output.finalize(),
            raw=self._state.agent_runs,
        )
        _run_graph_end_hooks(self._deps, res)
        self._auto_update_context(res)
        return res

    def run_sync(self) -> Result[Output]:
        """Execute the graph to completion and return the final result."""
        graph = self.prepare_graph()

        cleanup = _install_target_hooks(self._deps)
        _run_graph_start_hooks(self._deps, self._state)

        try:
            if self._state and self._deps:
                _run_coro_sync(
                    graph.run(
                        state=self._state,
                        deps=self._deps,
                        inputs=None,
                    )
                )
        except Exception as e:
            _run_graph_error_hooks(self._deps, e, self._state)
            _run_error_hooks(self._deps, e)
            raise
        finally:
            cleanup()

        res = Result(
            self._state.output.finalize(),
            raw=self._state.agent_runs,
        )
        _run_graph_end_hooks(self._deps, res)
        self._auto_update_context(res)
        return res

    async def stream(self, *, exclude_none: bool = False) -> Stream[Output]:
        """Execute the graph with streaming nodes and return a Stream wrapper.

        The graph should contain stream steps that store their streams in
        `ctx.state.streams` during execution. After the graph runs
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
        _run_graph_start_hooks(self._deps, self._state)

        try:
            if self._state and self._deps:
                await graph.run(
                    state=self._state,
                    deps=self._deps,
                    inputs=None,
                )
        except Exception as e:
            _run_graph_error_hooks(self._deps, e, self._state)
            _run_error_hooks(self._deps, e)
            raise
        finally:
            cleanup()

        stream = Stream(
            _builder=self._state.output,
            _streams=self._state.streams,
            _field_mappings=self._state.stream_field_mappings,  # type: ignore[arg-type]
            _stream_contexts=self._state.stream_contexts,
            _exclude_none=exclude_none,
        )
        _run_graph_end_hooks(self._deps, stream)
        return stream

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


def _run_graph_start_hooks(
    deps: SemanticGraphDeps[Any, Any], state: SemanticGraphState[Any]
) -> None:
    hooks: GraphHooks | None = getattr(deps, "hooks", None)
    if hooks is None or hooks.on_run_start is None:
        return
    try:
        hooks.on_run_start(SimpleNamespace(state=state, deps=deps))
    except Exception:
        pass


def _run_graph_end_hooks(
    deps: SemanticGraphDeps[Any, Any], result: Any
) -> None:
    hooks: GraphHooks | None = getattr(deps, "hooks", None)
    if hooks is None or hooks.on_run_end is None:
        return
    try:
        hooks.on_run_end(result)
    except Exception:
        pass


def _run_graph_error_hooks(
    deps: SemanticGraphDeps[Any, Any],
    error: BaseException,
    state: SemanticGraphState[Any],
) -> None:
    hooks: GraphHooks | None = getattr(deps, "hooks", None)
    if hooks is None or hooks.on_error is None:
        return
    try:
        hooks.on_error(
            error,
            SimpleNamespace(state=state, deps=deps),
        )
    except Exception:
        pass
