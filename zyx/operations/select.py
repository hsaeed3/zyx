"""zyx.operations.select"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Sequence,
    TypeVar,
    overload,
    cast,
)

from pydantic import BaseModel

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
from .._processing._outputs import selection_output_model
from .._types import (
    ContextType,
    ModelParam,
    ToolType,
)
from ..result import Result
from ..stream import Stream


__all__ = ("aselect", "select")


Deps = TypeVar("Deps")
Output = TypeVar("Output")


_SELECT_SYSTEM_PROMPT = (
    "\n[INSTRUCTION]\n"
    "- Your ONLY task is to select the best option(s) from the provided choices.\n"
    "- You MUST respond with a value that matches the given structured output schema "
    "exactly (no explanations, no conversational text, no extra keys).\n"
    "- The available choices are described in the schema docstring. Use them as the "
    "ground truth set of options.\n"
    "- Do NOT invent new options or modify the option values.\n"
    "- For single selection, choose exactly one option. For multi-selection, choose "
    "zero or more options.\n"
    "- Never return the literal value null as the entire response.\n"
)


def _normalize_choices(target: Any | Sequence[Any]) -> list[Any]:
    """Normalize the user-provided `target` into a flat list of concrete choices.

    This mirrors the behaviour expected by `selection_output_model` and supports:
    - list[...] of values or types
    - Literal[...]
    - Enum subclasses
    - Union[..., ...]
    """
    from enum import Enum
    from typing import get_args, get_origin

    # Already a concrete list of choices
    if isinstance(target, list):
        return list(target)

    origin = get_origin(target)
    args = get_args(target)

    # Enum class
    if isinstance(target, type) and issubclass(target, Enum):
        return list(target)

    # Literal[...] or Union[...] or list[T]
    if origin is not None:
        # list[T] – treat each element type/value as a choice
        if origin is list:
            return list(args)
        # Literal[...] / Union[...]
        return list(args)

    # Fallback: single value/type treated as the only choice
    return [target]


def _selection_to_output(
    target: Any | Sequence[Any],
    output: BaseModel,
    *,
    multi_select: bool = False,
    literal: bool = False,
) -> Any:
    """Map a structured selection model back to the original target object(s).

    The return value is:
    - A single selected object when `multi_select` is False
    - A list of selected objects when `multi_select` is True
    """

    # Try to re-use the choices stored on the model built by `selection_output_model`
    model_cls = type(output)
    choices: list[Any] | None = getattr(model_cls, "_choices", None)
    if choices is None:
        choices = _normalize_choices(target)

    def _select_one(idx_or_value: Any) -> Any:
        # Integer index into the choices list
        if isinstance(idx_or_value, int):
            if not choices:
                raise ValueError("No choices available for selection.")
            # Clamp the index defensively rather than raising IndexError
            if idx_or_value < 0 or idx_or_value >= len(choices):
                idx = max(0, min(len(choices) - 1, idx_or_value))
            else:
                idx = idx_or_value
            return choices[idx]

        # Literal mode or direct value – match by equality against choices
        for choice in choices:
            if choice == idx_or_value:
                return choice
        # If nothing matches, return the raw value as-is
        return idx_or_value

    if multi_select:
        indices: Iterable[Any] | None = getattr(output, "indices", None)
        if indices is None:
            # Graceful fallback – treat missing indices as empty selection
            return []
        return [_select_one(value) for value in indices]

    index_value: Any = getattr(output, "index", None)
    if index_value is None:
        # No explicit selection – return None
        return None
    return _select_one(index_value)


def prepare_select_graph(
    deps: SemanticGraphDeps[Deps, BaseModel],
    state: SemanticGraphState[BaseModel],
    *,
    multi_select: bool = False,
    literal: bool = False,
    include_reason: bool = False,
    stream: bool = False,
) -> SemanticGraph[BaseModel]:
    """
    Prepare a `SemanticGraph` for the `select` operation.

    The graph always generates an intermediate structured selection model; the
    public `select` / `aselect` helpers then map this back to the actual
    selected object(s).
    """
    # The deps.target has already been set to the selection model type.
    selection_model_type = cast(type[BaseModel], deps.target)

    if not deps.confidence:
        request = SemanticGraphRequestTemplate(
            output_type=selection_model_type,
            system_prompt_additions=_SELECT_SYSTEM_PROMPT,
            native_output=False if deps.toolsets else True,
        )
    else:
        request = SemanticGraphRequestTemplate(
            output_type=selection_model_type,
            system_prompt_additions=_SELECT_SYSTEM_PROMPT,
            native_output=True,
        )

    if stream:
        nodes = [SemanticStreamNode]
        start_node = SemanticStreamNode(request=request)
    else:
        nodes = [SemanticGenerateNode]
        start_node = SemanticGenerateNode(request=request)

    return SemanticGraph(nodes=nodes, start=start_node, state=state, deps=deps)


@dataclass
class _SelectionStreamWrapper:
    """Lightweight wrapper that maps the final streamed selection model back
    to the actual selected object(s), while preserving the underlying stream
    behaviour for text/partial/field.
    """

    _inner: Stream[BaseModel]
    _target: Any | Sequence[Any]
    _multi_select: bool
    _literal: bool

    async def __aenter__(self) -> "_SelectionStreamWrapper":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._inner.__aexit__(exc_type, exc_val, exc_tb)

    # --- Streaming primitives delegate to the inner stream unchanged ---

    def text(self, *, delta: bool = False, debounce_by: float | None = 0.1):
        return self._inner.text(delta=delta, debounce_by=debounce_by)

    def partial(self, *, debounce_by: float | None = 0.1):
        return self._inner.partial(debounce_by=debounce_by)

    def field(self, field_name: str, *, debounce_by: float | None = 0.1):
        return self._inner.field(
            field_name=field_name, debounce_by=debounce_by
        )

    @property
    def is_streaming(self) -> bool:
        return self._inner.is_streaming

    @property
    def usage(self):
        return self._inner.usage

    # --- Finalisation helpers that remap the output ---

    async def finish_async(self) -> Result[Output]:
        base_result = await self._inner.finish_async()
        selected = _selection_to_output(
            self._target,
            base_result.output,
            multi_select=self._multi_select,
            literal=self._literal,
        )
        return Result(output=selected, raw=base_result.raw)

    def finish(self) -> Result[Output]:
        # Synchronous wrapper around `finish_async`, mirroring `Stream.finish`.
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.finish_async())


@overload
async def aselect(
    target: Any | List[Any],
    context: ContextType | List[ContextType] | None = ...,
    *,
    multi_select: bool = ...,
    literal: bool = ...,
    include_reason: bool = ...,
    confidence: bool = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    stream: Literal[True],
) -> _SelectionStreamWrapper: ...


@overload
async def aselect(
    target: Any | List[Any],
    context: ContextType | List[ContextType] | None = ...,
    *,
    multi_select: bool = ...,
    literal: bool = ...,
    include_reason: bool = ...,
    confidence: bool = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    stream: Literal[False] = False,
) -> Result[Output]: ...


async def aselect(
    target: Any | List[Any],
    context: ContextType | List[ContextType] | None = None,
    *,
    multi_select: bool = False,
    literal: bool = False,
    include_reason: bool = False,
    confidence: bool = False,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: PydanticAIModelSettings | None = None,
    instructions: PydanticAIInstructions | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    usage_limits: PydanticAIUsageLimits | None = None,
    stream: bool = False,
) -> Result[Output] | _SelectionStreamWrapper:
    """Asynchronously select one or more options from the given `target`.

    Args:
        target (Any | List[Any]): The options to select from. Can be:
            - A list of values or types
            - A Literal[...] type
            - An Enum subclass
            - A Union[..., ...] type
        context (ContextType | List[ContextType] | None): Optional additional context or conversation history. Defaults to None.
        multi_select (bool): When True, return a list of selected objects; otherwise
            return a single selected object. Defaults to False.
        literal (bool): When True and supported by the options, use Literal[...] for
            the selection field instead of indices. Defaults to False.
        include_reason (bool): Reserved flag to include a textual reason field in the
            intermediate structured output (not exposed in the final result). Defaults to False.
        confidence (bool): When True, enable log-probability based confidence scoring. Defaults to False.
        model (ModelParam): The model to use for selection. Defaults to "openai:gpt-4o-mini".
        model_settings (PydanticAIModelSettings | None): Model settings to pass to the operation (e.g., temperature). Defaults to None.
        instructions (PydanticAIInstructions | None): Additional instructions/hints for the model. Defaults to None.
        tools (ToolType | List[ToolType] | None): List of tools available to the model. Defaults to None.
        deps (Deps | None): Optional dependencies (e.g., `pydantic_ai.RunContext`) for this operation. Defaults to None.
        usage_limits (PydanticAIUsageLimits | None): Usage limits (token/request) configuration. Defaults to None.
        stream (bool): When True, return a stream wrapper that exposes the underlying
            model stream and remaps the final result to the selected object(s). Defaults to False.

    Returns:
        Result[Output] | _SelectionStreamWrapper: A `Result` whose `output` is either the selected object or a list
            of selected objects, or a streaming wrapper with the same final
            semantics.
    """
    # Build the structured selection model for these options.
    selection_model = selection_output_model(
        options=target,
        name="Selection",
        multi_select=multi_select,
        literal=literal,
        reason=include_reason,
    )

    graph_deps = SemanticGraphDeps.prepare(
        model=model,
        model_settings=model_settings,
        context=context,
        instructions=instructions,  # type: ignore[arg-type]
        tools=tools,
        deps=deps,
        usage_limits=usage_limits,
        target=selection_model,
        source=target,
        confidence=confidence,
    )
    graph_state = SemanticGraphState.prepare(deps=graph_deps)

    graph = prepare_select_graph(
        deps=graph_deps,
        state=graph_state,
        multi_select=multi_select,
        literal=literal,
        include_reason=include_reason,
        stream=stream,
    )

    if stream:
        inner_stream: Stream[BaseModel] = await graph.stream()
        return _SelectionStreamWrapper(
            _inner=inner_stream,
            _target=target,
            _multi_select=multi_select,
            _literal=literal,
        )

    base_result = await graph.run()
    selected = _selection_to_output(
        target,
        base_result.output,
        multi_select=multi_select,
        literal=literal,
    )
    return Result(output=selected, raw=base_result.raw)


@overload
def select(
    target: Any | List[Any],
    context: ContextType | List[ContextType] | None = ...,
    *,
    multi_select: bool = ...,
    literal: bool = ...,
    include_reason: bool = ...,
    confidence: bool = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    stream: Literal[True],
) -> _SelectionStreamWrapper: ...


@overload
def select(
    target: Any | List[Any],
    context: ContextType | List[ContextType] | None = ...,
    *,
    multi_select: bool = ...,
    literal: bool = ...,
    include_reason: bool = ...,
    confidence: bool = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    stream: Literal[False] = False,
) -> Result[Output]: ...


def select(
    target: Any | List[Any],
    context: ContextType | List[ContextType] | None = None,
    *,
    multi_select: bool = False,
    literal: bool = False,
    include_reason: bool = False,
    confidence: bool = False,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: PydanticAIModelSettings | None = None,
    instructions: PydanticAIInstructions | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    usage_limits: PydanticAIUsageLimits | None = None,
    stream: bool = False,
) -> Result[Output] | _SelectionStreamWrapper:
    """Synchronously select one or more options from the given `target`.

    Args:
        target (Any | List[Any]): The options to select from. Can be:
            - A list of values or types
            - A Literal[...] type
            - An Enum subclass
            - A Union[..., ...] type
        context (ContextType | List[ContextType] | None): Optional additional context or conversation history. Defaults to None.
        multi_select (bool): When True, return a list of selected objects; otherwise
            return a single selected object. Defaults to False.
        literal (bool): When True and supported by the options, use Literal[...] for
            the selection field instead of indices. Defaults to False.
        include_reason (bool): Reserved flag to include a textual reason field in the
            intermediate structured output (not exposed in the final result). Defaults to False.
        confidence (bool): When True, enable log-probability based confidence scoring. Defaults to False.
        model (ModelParam): The model to use for selection. Defaults to "openai:gpt-4o-mini".
        model_settings (PydanticAIModelSettings | None): Model settings to pass to the operation (e.g., temperature). Defaults to None.
        instructions (PydanticAIInstructions | None): Additional instructions/hints for the model. Defaults to None.
        tools (ToolType | List[ToolType] | None): List of tools available to the model. Defaults to None.
        deps (Deps | None): Optional dependencies (e.g., `pydantic_ai.RunContext`) for this operation. Defaults to None.
        usage_limits (PydanticAIUsageLimits | None): Usage limits (token/request) configuration. Defaults to None.
        stream (bool): When True, return a stream wrapper that exposes the underlying
            model stream and remaps the final result to the selected object(s). Defaults to False.

    Returns:
        Result[Output] | _SelectionStreamWrapper: A `Result` whose `output` is either the selected object or a list
            of selected objects, or a streaming wrapper with the same final
            semantics.
    """
    # Build the structured selection model for these options.
    selection_model = selection_output_model(
        options=target,
        name="Selection",
        multi_select=multi_select,
        literal=literal,
        reason=include_reason,
    )

    graph_deps = SemanticGraphDeps.prepare(
        model=model,
        model_settings=model_settings,
        context=context,
        instructions=instructions,  # type: ignore[arg-type]
        tools=tools,
        deps=deps,
        usage_limits=usage_limits,
        target=selection_model,
        source=target,
        confidence=confidence,
    )
    graph_state = SemanticGraphState.prepare(deps=graph_deps)

    graph = prepare_select_graph(
        deps=graph_deps,
        state=graph_state,
        multi_select=multi_select,
        literal=literal,
        include_reason=include_reason,
        stream=stream,
    )

    if stream:
        inner_stream: Stream[BaseModel] = graph.stream_sync()
        return _SelectionStreamWrapper(
            _inner=inner_stream,
            _target=target,
            _multi_select=multi_select,
            _literal=literal,
        )

    base_result = graph.run_sync()
    selected = _selection_to_output(
        target,
        base_result.output,
        multi_select=multi_select,
        literal=literal,
    )
    return Result(output=selected, raw=base_result.raw)
