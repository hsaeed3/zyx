"""zyx.operations.edit"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Literal, TypeVar, overload

from ..._aliases import (
    PydanticAIInstructions,
    PydanticAIModelSettings,
    PydanticAIUsageLimits,
)
from ..._graph import (
    AbstractSemanticNode,
    SemanticGraph,
    SemanticGraphDeps,
    SemanticGraphState,
    SemanticGraphRequestTemplate,
    SemanticGraphContext,
    End,
)
from ..._processing._outputs import split_output_model_by_fields
from ..._types import (
    ModelParam,
    SourceParam,
    TargetParam,
    ContextType,
    ToolType,
    AttachmentType,
)
from ...result import Result
from ...stream import Stream
from .strategy import (
    EditStrategy,
    AbstractEditStrategy,
    TextEditStrategy,
)


Deps = TypeVar("Deps")
Output = TypeVar("Output")

_logger = logging.getLogger("zyx.operations.edit")

_EDIT_SYSTEM_PROMPT = (
    "\n[INSTRUCTION]\n"
    "- You are an editor. Your task is to edit the PRIMARY INPUT to satisfy the user instructions.\n"
    "- Return ONLY the structured output that matches the provided schema.\n"
    "- If no edits are necessary, return the original content unchanged in the schema format.\n"
    "- For selective edits, return the final updated values for any edited fields (not deltas or fragments).\n"
)

_EDIT_PLAN_PROMPT = (
    "\n[INSTRUCTION]\n"
    "- You are planning edits. Determine which parts should be edited.\n"
    "- Return ONLY the plan schema. Do NOT perform the edits yet.\n"
    "- If no edits are needed, choose the null/no-op option.\n"
)


def _native_output_for_edit(deps: SemanticGraphDeps[Deps, Output]) -> bool:
    if deps.confidence:
        return True
    return False if deps.toolsets else True


def _plan_has_edits(strategy: AbstractEditStrategy[Output], plan: Any) -> bool:
    if strategy.kind == "mapping":
        selections = getattr(strategy, "_extract_plan_selections", None)
        if selections is None:
            return False
        return bool(selections(plan))
    if strategy.kind == "text":
        selections = getattr(plan, "selections", None)
        if selections in (None, "null"):
            return False
        return isinstance(selections, list) and len(selections) > 0
    if strategy.kind == "basic":
        return getattr(plan, "selection", None) != "null"
    return True


@dataclass
class ReplaceEditNode(AbstractSemanticNode[Deps, Output]):
    """
    One-off node responsible for generating a complete replacement/edit for the
    target object when using the `edit` semantic operation.

    This is only used when `selective` is set to False.
    """

    request: SemanticGraphRequestTemplate[Output]
    strategy: AbstractEditStrategy[Output]
    exclude_none: bool = False

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output]:
        result = await self.execute_run(
            ctx=ctx, request=self.request, update_output=False
        )
        self.strategy.apply_replacement(result.output)
        return End(ctx.state.output.finalize(exclude_none=self.exclude_none))


@dataclass
class ReplaceEditStreamNode(AbstractSemanticNode[Deps, Output]):
    request: SemanticGraphRequestTemplate[Output]
    update_output: bool = True
    output_fields: str | List[str] | None = None

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output]:
        stream_ctx = self.execute_stream(ctx=ctx, request=self.request)
        stream = await stream_ctx.__aenter__()
        ctx.state.streams.append(stream)  # type: ignore
        ctx.state.stream_contexts.append(stream_ctx)
        ctx.state.stream_field_mappings.append(
            {
                "stream_index": len(ctx.state.streams) - 1,
                "fields": self.output_fields
                if isinstance(self.output_fields, list)
                else [self.output_fields]
                if self.output_fields
                else None,
                "update_output": self.update_output,
            }
        )
        return End(None)  # type: ignore[return-value]


@dataclass
class SelectiveEditNode(AbstractSemanticNode[Deps, Output]):
    """
    Node responsible for generating a one-off edit for the target object, while
    using the `edit` semantic operation with `selective` set to True.
    """

    request: SemanticGraphRequestTemplate[Output]
    strategy: AbstractEditStrategy[Output]
    exclude_none: bool = False

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output]:
        result = await self.execute_run(
            ctx=ctx, request=self.request, update_output=False
        )
        self.strategy.apply_selective(result.output)
        return End(ctx.state.output.finalize(exclude_none=self.exclude_none))


@dataclass
class PlanEditNode(AbstractSemanticNode[Deps, Output]):
    request: SemanticGraphRequestTemplate[Output]
    strategy: AbstractEditStrategy[Output]
    iterative: bool
    stream: bool
    exclude_none: bool = False

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output] | AbstractSemanticNode[Deps, Output]:
        result = await self.execute_run(
            ctx=ctx, request=self.request, update_output=False
        )
        plan = result.output
        setattr(ctx.state, "edit_plan", plan)

        if not _plan_has_edits(self.strategy, plan):
            return End(
                ctx.state.output.finalize(exclude_none=self.exclude_none)
            )

        if self.iterative:
            if self.stream:
                return PlanIterativeStreamNode(
                    base_request=_build_edit_request(
                        ctx,
                        output_type=None,
                        system_prompt_additions=_EDIT_SYSTEM_PROMPT,
                    ),
                    strategy=self.strategy,
                    exclude_none=self.exclude_none,
                )
            return PlanIterativeEditNode(
                base_request=_build_edit_request(
                    ctx,
                    output_type=None,
                    system_prompt_additions=_EDIT_SYSTEM_PROMPT,
                ),
                strategy=self.strategy,
                exclude_none=self.exclude_none,
            )

        if self.stream:
            return PlanEditsStreamNode(
                request=_build_edit_request(
                    ctx,
                    output_type=self.strategy.get_plan_edit_schema(plan),
                    system_prompt_additions=_EDIT_SYSTEM_PROMPT,
                ),
                exclude_none=self.exclude_none,
            )

        return PlanEditsNode(
            request=_build_edit_request(
                ctx,
                output_type=self.strategy.get_plan_edit_schema(plan),
                system_prompt_additions=_EDIT_SYSTEM_PROMPT,
            ),
            strategy=self.strategy,
            exclude_none=self.exclude_none,
        )


@dataclass
class PlanEditsNode(AbstractSemanticNode[Deps, Output]):
    request: SemanticGraphRequestTemplate[Output]
    strategy: AbstractEditStrategy[Output]
    exclude_none: bool = False

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output]:
        plan = getattr(ctx.state, "edit_plan", None)
        result = await self.execute_run(
            ctx=ctx, request=self.request, update_output=False
        )
        if plan is None:
            self.strategy.apply_replacement(result.output)
        else:
            self.strategy.apply_plan_edits(plan, result.output)
        return End(ctx.state.output.finalize(exclude_none=self.exclude_none))


@dataclass
class PlanEditsStreamNode(AbstractSemanticNode[Deps, Output]):
    request: SemanticGraphRequestTemplate[Output]
    exclude_none: bool = False

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output]:
        stream_ctx = self.execute_stream(ctx=ctx, request=self.request)
        stream = await stream_ctx.__aenter__()
        ctx.state.streams.append(stream)  # type: ignore
        ctx.state.stream_contexts.append(stream_ctx)
        ctx.state.stream_field_mappings.append(
            {
                "stream_index": len(ctx.state.streams) - 1,
                "fields": None,
                "update_output": True,
            }
        )
        return End(None)  # type: ignore[return-value]


@dataclass
class PlanIterativeEditNode(AbstractSemanticNode[Deps, Output]):
    base_request: SemanticGraphRequestTemplate[Output]
    strategy: AbstractEditStrategy[Output]
    exclude_none: bool = False

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output] | AbstractSemanticNode[Deps, Output]:
        plan = getattr(ctx.state, "edit_plan", None)
        if plan is None:
            return End(
                ctx.state.output.finalize(exclude_none=self.exclude_none)
            )

        items = getattr(ctx.state, "edit_items", None)
        if items is None:
            items = _build_iterative_items(self.strategy, plan)
            setattr(ctx.state, "edit_items", items)
            setattr(ctx.state, "edit_index", 0)

        idx = getattr(ctx.state, "edit_index", 0)
        if idx >= len(items):
            return End(
                ctx.state.output.finalize(exclude_none=self.exclude_none)
            )

        item = items[idx]
        request = _build_edit_request(
            ctx,
            output_type=item["schema"],
            system_prompt_additions=_EDIT_SYSTEM_PROMPT,
        )
        result = await self.execute_run(
            ctx=ctx, request=request, update_output=False
        )

        if self.strategy.kind == "mapping":
            field = item.get("field")
            if field:
                ctx.state.output.update_from_pydantic_ai_result(
                    result,
                    fields=field,
                )
            else:
                self.strategy.apply_replacement(result.output)
        elif self.strategy.kind == "text":
            if isinstance(self.strategy, TextEditStrategy):
                self.strategy.apply_plan_edit_at_index(
                    plan, idx, result.output
                )
            else:
                self.strategy.apply_replacement(result.output)
        else:
            self.strategy.apply_replacement(result.output)

        setattr(ctx.state, "edit_index", idx + 1)
        return self


@dataclass
class PlanIterativeStreamNode(AbstractSemanticNode[Deps, Output]):
    base_request: SemanticGraphRequestTemplate[Output]
    strategy: AbstractEditStrategy[Output]
    exclude_none: bool = False

    async def run(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> End[Output] | AbstractSemanticNode[Deps, Output]:
        if self.strategy.kind != "mapping":
            raise ValueError(
                "Iterative streaming edits are only supported for mapping targets."
            )

        plan = getattr(ctx.state, "edit_plan", None)
        if plan is None:
            return End(None)  # type: ignore[return-value]

        items = getattr(ctx.state, "edit_items", None)
        if items is None:
            items = _build_iterative_items(self.strategy, plan)
            setattr(ctx.state, "edit_items", items)
            setattr(ctx.state, "edit_index", 0)

        idx = getattr(ctx.state, "edit_index", 0)
        if idx >= len(items):
            return End(None)  # type: ignore[return-value]

        item = items[idx]
        request = _build_edit_request(
            ctx,
            output_type=item["schema"],
            system_prompt_additions=_EDIT_SYSTEM_PROMPT,
        )
        stream_ctx = self.execute_stream(ctx=ctx, request=request)
        stream = await stream_ctx.__aenter__()
        ctx.state.streams.append(stream)  # type: ignore
        ctx.state.stream_contexts.append(stream_ctx)
        ctx.state.stream_field_mappings.append(
            {
                "stream_index": len(ctx.state.streams) - 1,
                "fields": [item.get("field")] if item.get("field") else None,
                "update_output": True,
            }
        )
        setattr(ctx.state, "edit_index", idx + 1)
        return self


def _build_iterative_items(
    strategy: AbstractEditStrategy[Output], plan: Any
) -> List[dict[str, Any]]:
    items: List[dict[str, Any]] = []
    if strategy.kind == "mapping":
        selections = getattr(strategy, "_extract_plan_selections")(plan)
        split = split_output_model_by_fields(strategy.builder.normalized)
        for field in selections:
            model = split.get(field)
            if model is None:
                continue
            items.append({"field": field, "schema": model})
        return items
    if strategy.kind == "text":
        selections = getattr(plan, "selections", None)
        if selections in (None, "null"):
            return items
        for selection in selections:
            items.append(
                {
                    "selection": selection,
                    "schema": strategy.get_replacement_schema(),
                }
            )
        return items
    if strategy.kind == "basic":
        if getattr(plan, "selection", None) == "replace":
            items.append({"schema": strategy.get_replacement_schema()})
    return items


def _build_edit_request(
    ctx: SemanticGraphContext[Output, Deps],
    *,
    output_type: Any,
    system_prompt_additions: str | None,
) -> SemanticGraphRequestTemplate[Output]:
    return SemanticGraphRequestTemplate(
        output_type=output_type,
        system_prompt_additions=system_prompt_additions,
        include_output_context=True,
        native_output=_native_output_for_edit(ctx.deps),
    )


def prepare_edit_graph(
    deps: SemanticGraphDeps[Deps, Output],
    state: SemanticGraphState[Output],
    *,
    selective: bool,
    iterative: bool,
    plan: bool,
    merge: bool,
    stream: bool,
) -> SemanticGraph[Output]:
    strategy: AbstractEditStrategy[Output] = EditStrategy.create(state.output)

    if merge and strategy.kind != "mapping":
        raise ValueError(
            "`merge=True` is only supported for mapping-like targets."
        )

    if merge and not (selective or plan):
        raise ValueError(
            "`merge=True` requires `selective=True` or `plan=True`."
        )

    if merge and not (selective or plan):
        raise ValueError(
            "`merge=True` requires `selective=True` or `plan=True`."
        )

    if iterative and not plan:
        raise ValueError("`iterative=True` requires `plan=True`.")

    if stream and strategy.kind == "text" and (selective or plan):
        raise ValueError(
            "Streaming selective/plan edits are not supported for text targets."
        )

    if stream and merge and not (plan and iterative):
        raise ValueError(
            "`merge=True` with streaming requires `plan=True` and `iterative=True`."
        )

    if stream and merge and strategy.kind != "mapping":
        raise ValueError(
            "`merge=True` streaming edits are only supported for mapping targets."
        )

    exclude_none = False

    if plan:
        plan_request = SemanticGraphRequestTemplate(
            output_type=strategy.get_plan_schema(),
            system_prompt_additions=_EDIT_PLAN_PROMPT,
            include_output_context=True,
            native_output=_native_output_for_edit(deps),
        )
        start_node: PlanEditNode[Any, Any] = PlanEditNode(
            request=plan_request,
            strategy=strategy,
            iterative=iterative,
            stream=stream,
            exclude_none=exclude_none,
        )
        nodes = [
            PlanEditNode,
            PlanEditsNode,
            PlanEditsStreamNode,
            PlanIterativeEditNode,
            PlanIterativeStreamNode,
        ]
        return SemanticGraph(
            nodes=nodes,
            start=start_node,
            state=state,
            deps=deps,
        )

    if selective:
        if stream:
            # Streaming selective edits fallback to full replacement output.
            request = SemanticGraphRequestTemplate(
                output_type=strategy.get_replacement_schema(),
                system_prompt_additions=_EDIT_SYSTEM_PROMPT,
                include_output_context=True,
                native_output=_native_output_for_edit(deps),
            )
            start_node = ReplaceEditStreamNode(request=request)
            nodes = [ReplaceEditStreamNode]
            return SemanticGraph(
                nodes=nodes,
                start=start_node,  # type: ignore[arg-type]
                state=state,  # type: ignore[arg-type]
                deps=deps,  # type: ignore
            )

        request = SemanticGraphRequestTemplate(
            output_type=strategy.get_selective_schema(),
            system_prompt_additions=_EDIT_SYSTEM_PROMPT,
            include_output_context=True,
            native_output=_native_output_for_edit(deps),
        )
        start_node = SelectiveEditNode(
            request=request, strategy=strategy, exclude_none=exclude_none
        )
        nodes = [SelectiveEditNode]
        return SemanticGraph(
            nodes=nodes,
            start=start_node,
            state=state,  # type: ignore[arg-type]
            deps=deps,  # type: ignore
        )

    if stream:
        request = SemanticGraphRequestTemplate(
            output_type=strategy.get_replacement_schema(),
            system_prompt_additions=_EDIT_SYSTEM_PROMPT,
            include_output_context=True,
            native_output=_native_output_for_edit(deps),
        )
        start_node = ReplaceEditStreamNode(request=request)
        nodes = [ReplaceEditStreamNode]
        return SemanticGraph(
            nodes=nodes,
            start=start_node,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
            deps=deps,  # type: ignore
        )

    request = SemanticGraphRequestTemplate(
        output_type=strategy.get_replacement_schema(),
        system_prompt_additions=_EDIT_SYSTEM_PROMPT,
        include_output_context=True,
        native_output=_native_output_for_edit(deps),
    )
    start_node = ReplaceEditNode(
        request=request, strategy=strategy, exclude_none=exclude_none
    )
    nodes = [ReplaceEditNode]
    return SemanticGraph(nodes=nodes, start=start_node, state=state, deps=deps)  # type: ignore


@overload
async def aedit(
    target: TargetParam[Output],
    context: ContextType | List[ContextType] | None = ...,
    *,
    selective: bool = ...,
    iterative: bool = ...,
    plan: bool = ...,
    merge: bool = ...,
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
async def aedit(
    target: TargetParam[Output],
    context: ContextType | List[ContextType] | None = ...,
    *,
    selective: bool = ...,
    iterative: bool = ...,
    plan: bool = ...,
    merge: bool = ...,
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


async def aedit(
    target: TargetParam[Output],
    context: ContextType | List[ContextType] | None = None,
    *,
    selective: bool = True,
    iterative: bool = False,
    plan: bool = False,
    merge: bool = False,
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
    """Asynchronously edit a target value using a model or Pydantic AI agent.

    Args:
        target (TargetParam[Output]): The target value or type to edit.
        context (ContextType | List[ContextType] | None): Optional context or conversation history for the operation. Defaults to None.
        selective (bool): When True, perform selective edits. Defaults to True.
        iterative (bool): When True, perform iterative edits. Requires plan=True. Defaults to False.
        plan (bool): When True, create an edit plan first. Defaults to False.
        merge (bool): When True, merge edits with existing values. Requires selective=True or plan=True. Defaults to False.
        confidence (bool): When True, enables log-probability based confidence scoring. Defaults to False.
        model (ModelParam): The model to use for editing. Can be a string, Pydantic AI model, or agent. Defaults to "openai:gpt-4o-mini".
        model_settings (PydanticAIModelSettings | None): Model settings to pass to the operation (e.g., temperature). Defaults to None.
        attachments (AttachmentType | List[AttachmentType] | None): Attachments, e.g. `Snippet` or `AbstractResource`, provided to the agent. Defaults to None.
        instructions (PydanticAIInstructions | None): Additional instructions/hints for the model. Defaults to None.
        tools (ToolType | List[ToolType] | None): List of tools available to the model. Defaults to None.
        deps (Deps | None): Optional dependencies (e.g., `pydantic_ai.RunContext`) for this operation. Defaults to None.
        usage_limits (PydanticAIUsageLimits | None): Usage limits (token/request) configuration. Defaults to None.
        stream (bool): Whether to stream the output of the operation. Defaults to False.

    Returns:
        Result[Output] | Stream[Output]: Edited result or stream of outputs, depending on `stream`.
    """
    from ...targets import Target as TargetClass

    _target = target
    _instructions = instructions
    _source: SourceParam | None = None

    if isinstance(target, TargetClass):
        _target = target.target
        if target.instructions and not _instructions:
            _instructions = target.instructions

    if not isinstance(_target, type):
        _source = _target

    graph_deps = SemanticGraphDeps.prepare(
        model=model,
        model_settings=model_settings,
        context=context,
        instructions=_instructions,  # type: ignore[arg-type]
        tools=tools,
        deps=deps,
        attachments=attachments,
        usage_limits=usage_limits,
        target=_target,
        source=_source,
        confidence=confidence,
    )
    graph_state = SemanticGraphState.prepare(deps=graph_deps)

    strategy = EditStrategy.create(graph_state.output)
    if merge and strategy.kind != "mapping":
        raise ValueError(
            "`merge=True` is only supported for mapping-like targets."
        )

    # Seed for full replacements and for merge-mode selective/plan edits.
    if not isinstance(_target, type) and (not selective or merge):
        graph_state.output.update(_target)

    graph = prepare_edit_graph(
        deps=graph_deps,
        state=graph_state,
        selective=selective,
        iterative=iterative,
        plan=plan,
        merge=merge,
        stream=stream,
    )

    if stream:
        return await graph.stream(exclude_none=False)
    return await graph.run()


@overload
def edit(
    target: TargetParam[Output],
    context: ContextType | List[ContextType] | None = ...,
    *,
    selective: bool = ...,
    iterative: bool = ...,
    plan: bool = ...,
    merge: bool = ...,
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
def edit(
    target: TargetParam[Output],
    context: ContextType | List[ContextType] | None = ...,
    *,
    selective: bool = ...,
    iterative: bool = ...,
    plan: bool = ...,
    merge: bool = ...,
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


def edit(
    target: TargetParam[Output],
    context: ContextType | List[ContextType] | None = None,
    *,
    selective: bool = True,
    iterative: bool = False,
    plan: bool = False,
    merge: bool = False,
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
    """Synchronously edit a target value using a model or Pydantic AI agent.

    Args:
        target (TargetParam[Output]): The target value or type to edit.
        context (ContextType | List[ContextType] | None): Optional context or conversation history for the operation. Defaults to None.
        selective (bool): When True, perform selective edits. Defaults to True.
        iterative (bool): When True, perform iterative edits. Requires plan=True. Defaults to False.
        plan (bool): When True, create an edit plan first. Defaults to False.
        merge (bool): When True, merge edits with existing values. Requires selective=True or plan=True. Defaults to False.
        confidence (bool): When True, enables log-probability based confidence scoring. Defaults to False.
        model (ModelParam): The model to use for editing. Can be a string, Pydantic AI model, or agent. Defaults to "openai:gpt-4o-mini".
        model_settings (PydanticAIModelSettings | None): Model settings to pass to the operation (e.g., temperature). Defaults to None.
        attachments (AttachmentType | List[AttachmentType] | None): Attachments, e.g. `Snippet` or `AbstractResource`, provided to the agent. Defaults to None.
        instructions (PydanticAIInstructions | None): Additional instructions/hints for the model. Defaults to None.
        tools (ToolType | List[ToolType] | None): List of tools available to the model. Defaults to None.
        deps (Deps | None): Optional dependencies (e.g., `pydantic_ai.RunContext`) for this operation. Defaults to None.
        usage_limits (PydanticAIUsageLimits | None): Usage limits (token/request) configuration. Defaults to None.
        stream (bool): Whether to stream the output of the operation. Defaults to False.

    Returns:
        Result[Output] | Stream[Output]: Edited result or stream of outputs, depending on `stream`.
    """
    from ...targets import Target as TargetClass

    _target = target
    _instructions = instructions
    _source: SourceParam | None = None

    if isinstance(target, TargetClass):
        _target = target.target
        if target.instructions and not _instructions:
            _instructions = target.instructions

    if not isinstance(_target, type):
        _source = _target

    graph_deps = SemanticGraphDeps.prepare(
        model=model,
        model_settings=model_settings,
        context=context,
        instructions=_instructions,  # type: ignore[arg-type]
        tools=tools,
        deps=deps,
        attachments=attachments,
        usage_limits=usage_limits,
        target=_target,
        source=_source,
        confidence=confidence,
    )
    graph_state = SemanticGraphState.prepare(deps=graph_deps)

    strategy = EditStrategy.create(graph_state.output)
    if merge and strategy.kind != "mapping":
        raise ValueError(
            "`merge=True` is only supported for mapping-like targets."
        )

    if not isinstance(_target, type) and (not selective or merge):
        graph_state.output.update(_target)

    graph = prepare_edit_graph(
        deps=graph_deps,
        state=graph_state,
        selective=selective,
        iterative=iterative,
        plan=plan,
        merge=merge,
        stream=stream,
    )

    if stream:
        return graph.stream_sync(exclude_none=False)
    return graph.run_sync()
