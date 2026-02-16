"""zyx.operations.run"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    cast,
    overload,
    TYPE_CHECKING,
)

from pydantic import BaseModel
import pydantic_monty
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext as PydanticAIRunContext
from pydantic_ai.toolsets import FunctionToolset

from .._aliases import (
    PydanticAIInstructions,
    PydanticAIModelSettings,
    PydanticAIUsage,
    PydanticAIUsageLimits,
    PydanticAIModel,
)
from .._graph import SemanticGraphDeps, SemanticGraphRequestTemplate
from .._processing._toon import object_as_text
from .._types import (
    AttachmentType,
    ContextType,
    Deps,
    ModelParam,
    SourceParam,
    TargetParam,
    ToolType,
)
from .._utils._outputs import OutputBuilder
from .._utils._semantic import semantic_for_operation
from ..result import Result
from .make import amake
from ..stream import Stream, StreamFieldMapping as StreamStreamFieldMapping
from ..targets import Target

if TYPE_CHECKING:
    from .._utils._observer import Observer

__all__ = (
    "RunCompletion",
    "RunTask",
    "RunVerificationContext",
    "RunResult",
    "RunStream",
    "arun",
    "run",
)


Output = TypeVar("Output")


_RUN_SYSTEM_PROMPT = (
    "\n[INSTRUCTION]\n"
    "- You are executing a task. Use the available tools and attachments to complete it.\n"
    "- You MUST call the `complete_task` tool exactly once when the task is complete.\n"
    "- Do not claim completion in plain text; completion happens only via the tool call.\n"
    "- If you are missing information, use tools or ask for clarification (via tools if available).\n"
    "- Your completion will be deterministically verified. If verification fails, you must try again.\n"
)

_PLAN_SYSTEM_PROMPT = (
    "\n[INSTRUCTION]\n"
    "- Produce a deterministic execution plan for this task as pure Python code.\n"
    "- Output ONLY Python code. No prose, no JSON, no markdown fences.\n"
    "- Call tools directly by name, e.g. front_door_light(False).\n"
    '- Use `prompt("...")` to queue follow-up instructions.\n'
    "- Use `complete(...)` to finalize the task.\n"
    "- Do NOT use await, multi_tool_use, or functions.* prefixes.\n"
)


class RunCompletion(BaseModel):
    """Structured completion payload emitted by the task completion tool."""

    status: Literal["success", "failed", "skipped"] = "success"
    summary: str
    result: Any | None = None
    evidence: Any | None = None
    verification: Any | None = None


@dataclass
class RunTask(Generic[Output]):
    """Lightweight task metadata for a run."""

    goal: str | None
    target: TargetParam[Output] | None
    source: SourceParam | None


@dataclass
class RunVerificationContext(Generic[Output]):
    """Context passed to a verifier for deterministic validation."""

    target: TargetParam[Output] | None
    source: SourceParam | None
    deps: Deps | None
    attachments: List[AttachmentType] | None


@dataclass
class RunResult(Result[Optional[Output]]):
    """Result wrapper for run() with completion details."""

    completion: RunCompletion | None = None
    verified: bool | None = None
    verification_error: str | None = None
    task: RunTask[Output] | None = None

    def __repr__(self) -> str:
        completion = self.completion.status if self.completion else "None"
        task_goal = self.task.goal if self.task else None
        return (
            f"RunResult({type(self.output).__name__}):\n"
            f"{self.output}\n\n"
            f">>> Model: {self.model}\n"
            f">>> Completion: {completion}\n"
            f">>> Verified: {self.verified}\n"
            f">>> Task: {task_goal}\n"
        )

    def __rich__(self):
        from rich.console import RenderableType, Group
        from rich.rule import Rule
        from rich.text import Text

        renderables: list[RenderableType] = []

        renderables.append(
            Rule(
                title=f"✨ RunResult({type(self.output).__name__})",
                style="rule.line",
                align="left",
            )
        )

        output_text = f"{self.output}\n"
        renderables.append(Text.from_markup(f"[bold]{output_text}[/bold]"))

        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Type: {type(self.output).__name__}[/dim italic]"
            )
        )

        model = getattr(self, "model", None)
        if model:
            model_names = (
                model
                if isinstance(model, str)
                else (model[0] if len(model) == 1 else ", ".join(model))
            )
            label = (
                "Model"
                if (
                    isinstance(model_names, str)
                    or (hasattr(model, "__len__") and len(model) == 1)
                )
                else "Models"
            )
            renderables.append(
                Text.from_markup(
                    f"[sandy_brown]>>>[/sandy_brown] [dim italic]{label}: {model_names}[/dim italic]"
                )
            )

        completion = self.completion.status if self.completion else "None"
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Completion: {completion}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Verified: {self.verified}[/dim italic]"
            )
        )
        task_goal = self.task.goal if self.task else None
        if task_goal:
            renderables.append(
                Text.from_markup(
                    f"[sandy_brown]>>>[/sandy_brown] [dim italic]Task: {task_goal}[/dim italic]"
                )
            )

        return Group(*renderables)


@dataclass
class RunStream(Stream[Output]):
    completion: RunCompletion | None = None
    verified: bool | None = None
    verification_error: str | None = None
    task: RunTask[Output] | None = None
    _require_completion: bool = True
    _execution_state: RunExecutionState[Output] | None = None

    async def finish_async(self) -> RunResult[Output]:  # type: ignore[override]
        res = await super().finish_async()
        completion_state = self.completion
        if self._execution_state is not None:
            completion_state = (
                completion_state or self._execution_state.completion_state
            )
        if self._require_completion and completion_state is None:
            raise ValueError(
                "Task did not complete. The model must call `complete`."
            )
        completion_state = completion_state or RunCompletion(
            status="skipped",
            summary="No completion tool call was made.",
        )
        return RunResult(
            output=res.output,
            raw=res.raw,
            completion=completion_state,
            verified=self.verified
            if self._execution_state is None
            else self._execution_state.verified,
            verification_error=self.verification_error
            if self._execution_state is None
            else self._execution_state.verification_error,
            task=self.task,
        )

    def finish(self) -> RunResult[Output]:  # type: ignore[override]
        res = super().finish()
        completion_state = self.completion
        if self._execution_state is not None:
            completion_state = (
                completion_state or self._execution_state.completion_state
            )
        if self._require_completion and completion_state is None:
            raise ValueError(
                "Task did not complete. The model must call `complete`."
            )
        completion_state = completion_state or RunCompletion(
            status="skipped",
            summary="No completion tool call was made.",
        )
        return RunResult(
            output=res.output,
            raw=res.raw,
            completion=completion_state,
            verified=self.verified
            if self._execution_state is None
            else self._execution_state.verified,
            verification_error=self.verification_error
            if self._execution_state is None
            else self._execution_state.verification_error,
            task=self.task,
        )

    def __repr__(self) -> str:
        completion = self.completion.status if self.completion else "None"
        task_goal = self.task.goal if self.task else None
        return (
            f"RunStream({type(self._builder.normalized).__name__}):\n"
            f" >>> Is Streaming: {self.is_streaming}\n"
            f" >>> Is Complete: {self._is_complete}\n"
            f" >>> Completion: {completion}\n"
            f" >>> Verified: {self.verified}\n"
            f" >>> Task: {task_goal}\n"
        )

    def __rich__(self):
        from rich.console import RenderableType, Group
        from rich.rule import Rule
        from rich.text import Text

        renderables: list[RenderableType] = []

        renderables.append(
            Rule(title="✨ RunStream", style="rule.line", align="left")
        )

        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Type: {type(self._builder.normalized).__name__}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Is Streaming: {self.is_streaming}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Is Complete: {self._is_complete}[/dim italic]"
            )
        )

        completion = self.completion.status if self.completion else "None"
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Completion: {completion}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Verified: {self.verified}[/dim italic]"
            )
        )
        task_goal = self.task.goal if self.task else None
        if task_goal:
            renderables.append(
                Text.from_markup(
                    f"[sandy_brown]>>>[/sandy_brown] [dim italic]Task: {task_goal}[/dim italic]"
                )
            )

        if self._streams and len(self._streams) > 0:
            first_stream = self._streams[0]
            if (
                first_stream is not None
                and hasattr(first_stream, "response")
                and first_stream.response
                and first_stream.response.model_name
            ):
                renderables.append(
                    Text.from_markup(
                        f"[sandy_brown]>>>[/sandy_brown] [dim italic]Model: {first_stream.response.model_name}[/dim italic]"
                    )
                )

        return Group(*renderables)


@dataclass
class RunExecutionState(Generic[Output]):
    completion_state: RunCompletion | None = None
    validated_output: Output | None = None
    verified: bool | None = None
    verification_error: str | None = None
    tool_call_log: Dict[str, int] = field(default_factory=dict)
    total_tool_calls: int = 0


RunVerifier = (
    Callable[[RunCompletion], bool | str | None]
    | Callable[
        [RunCompletion, RunVerificationContext[Output]], bool | str | None
    ]
)

RunCheck = (
    Callable[[RunCompletion], bool | str | None]
    | Callable[
        [RunCompletion, RunVerificationContext[Output]], bool | str | None
    ]
)


def _build_target_schema_text(
    target: TargetParam[Output] | None,
) -> str | None:
    if target is None:
        return None

    if isinstance(target, Target):
        schema_target = target.target
        target_name = target.name
        target_desc = target.description
    else:
        schema_target = target
        target_name = None
        target_desc = None

    schema_text = object_as_text(schema_target)
    if target_name or target_desc:
        meta = []
        if target_name:
            meta.append(f"Name: {target_name}")
        if target_desc:
            meta.append(f"Description: {target_desc}")
        schema_text = "\n".join([*meta, "", schema_text])

    return schema_text


def _render_tool_inventory(toolsets: List[Any]) -> str | None:
    tools: List[str] = []
    for toolset_item in toolsets:
        if isinstance(toolset_item, FunctionToolset):
            for name, tool in toolset_item.tools.items():
                description = getattr(tool, "description", None) or ""
                if description:
                    tools.append(f"- {name}: {description}")
                else:
                    tools.append(f"- {name}")
    if not tools:
        return None
    return (
        "\n[AVAILABLE TOOLS]\n"
        + "\n".join(tools)
        + "\n[END AVAILABLE TOOLS]\n"
    )


def _observe_model_request(observe: Any | None) -> None:
    if observe and hasattr(observe, "on_model_request"):
        observe.on_model_request()


def _normalize_observer(observe: Any | None) -> Any | None:
    if not observe:
        return None
    from .._utils._observer import Observer

    if observe is True:
        return Observer()
    if isinstance(observe, Observer):
        return observe
    raise ValueError("Invalid 'observe' value. Expected bool or Observer.")


def _normalize_context(
    task: ContextType | List[ContextType] | None,
    context: ContextType | List[ContextType] | None,
) -> ContextType | List[ContextType] | None:
    if task is None:
        return context

    task_list = task if isinstance(task, list) else [task]
    if context is None:
        return task_list

    context_list = context if isinstance(context, list) else [context]
    return [*context_list, *task_list]  # type: ignore[return-value]


def _validate_target_result(
    target: TargetParam[Output] | None, result: Any
) -> Output | None:
    if target is None:
        return None

    if isinstance(target, Target):
        target_value = target.target
    else:
        target_value = target

    builder: OutputBuilder[Any] = OutputBuilder(target=target_value)
    builder.update(result)
    return builder.finalize()


def _evaluate_verifier(
    verifier: RunVerifier | None,
    completion: RunCompletion,
    context: RunVerificationContext[Output],
) -> tuple[bool | None, str | None]:
    if verifier is None:
        return None, None

    try:
        takes_ctx = len(inspect.signature(verifier).parameters) > 1
        if takes_ctx:
            verdict = verifier(completion, context)  # type: ignore[arg-type]
        else:
            verdict = verifier(completion)  # type: ignore[arg-type]
    except Exception as exc:
        return False, str(exc)

    if verdict is None or verdict is True:
        return True, None
    if verdict is False:
        return False, "Verification failed."
    if isinstance(verdict, str):
        return False, verdict
    return False, "Verification failed."


def _evaluate_checks(
    checks: List[RunCheck] | None,
    completion: RunCompletion,
    context: RunVerificationContext[Output],
) -> tuple[bool | None, str | None]:
    if not checks:
        return None, None
    for check in checks:
        try:
            takes_ctx = len(inspect.signature(check).parameters) > 1
            if takes_ctx:
                verdict = check(completion, context)  # type: ignore[arg-type]
            else:
                verdict = check(completion)  # type: ignore[arg-type]
        except Exception as exc:
            return False, str(exc)

        if verdict is None or verdict is True:
            continue
        if verdict is False:
            return False, "Final answer check failed."
        if isinstance(verdict, str):
            return False, verdict
    return True, None


def _normalize_attachments_list(
    attachments: AttachmentType | List[AttachmentType] | None,
) -> List[AttachmentType] | None:
    if attachments is None:
        return None
    if isinstance(attachments, list):
        return cast(List[AttachmentType], attachments)
    return [attachments]


def _apply_completion(
    *,
    state: RunExecutionState[Output],
    summary: str | None,
    status: Literal["success", "failed", "skipped"],
    result: Any | None,
    evidence: Any | None,
    verification: Any | None,
    target: TargetParam[Output] | None,
    source: SourceParam | None,
    deps: Deps | None,
    attachments: AttachmentType | List[AttachmentType] | None,
    verifier: RunVerifier | None,
    final_answer_checks: List[RunCheck] | None,
    error_cls: type[Exception],
) -> None:
    if target is not None and result is None:
        raise error_cls(
            "Task completion requires a result that matches the target schema."
        )

    if result is not None:
        if target is None:
            state.validated_output = result
        else:
            try:
                state.validated_output = _validate_target_result(
                    target, result
                )
            except Exception as exc:
                raise error_cls(
                    f"Task completion result failed validation: {exc}"
                ) from exc

    state.completion_state = RunCompletion(
        status=status,
        summary=summary or "Completed.",
        result=result,
        evidence=evidence,
        verification=verification,
    )

    verification_context = RunVerificationContext(
        target=target,
        source=source,
        deps=deps,  # type: ignore[arg-type]
        attachments=_normalize_attachments_list(attachments),
    )

    state.verified, state.verification_error = _evaluate_verifier(
        verifier, state.completion_state, verification_context
    )

    if state.verified is False:
        raise error_cls(
            state.verification_error or "Verification failed. Try again."
        )

    checks_ok, checks_error = _evaluate_checks(
        final_answer_checks, state.completion_state, verification_context
    )
    if checks_ok is False:
        raise error_cls(
            checks_error or "Final answer checks failed. Try again."
        )


async def _execute_plan_based_run(
    *,
    ctx: Any,
    graph_deps: SemanticGraphDeps[Deps, Output],
    run_state: SimpleNamespace,
    execution_state: RunExecutionState[Output],
    system_prompt: str,
    target: TargetParam[Output] | None,
    source: SourceParam | None,
    deps: Deps | None,
    attachments: AttachmentType | List[AttachmentType] | None,
    verifier: RunVerifier | None,
    max_retries: int,
    planning_interval: int | None,
    max_steps: int | None,
    final_answer_checks: List[RunCheck] | None,
    require_completion: bool,
) -> None:
    observe = getattr(graph_deps, "observe", None)
    tool_inventory = _render_tool_inventory(graph_deps.toolsets)
    if tool_inventory:
        system_prompt += tool_inventory

    plan_prompt = _PLAN_SYSTEM_PROMPT
    if planning_interval is not None:
        plan_prompt += (
            f"- Limit this plan to at most {planning_interval} tool calls.\n"
        )

    plan_rounds = max(1, max_retries)
    last_plan_error: str | None = None
    for round_idx in range(plan_rounds):
        plan_error: str | None = last_plan_error
        plan_code: str | None = None

        for _ in range(max_retries):
            retry_note = (
                f"\n[PLAN ERROR]\n{plan_error}\n[END PLAN ERROR]\n"
                if plan_error
                else ""
            )
            plan_request = SemanticGraphRequestTemplate(
                system_prompt_additions=system_prompt
                + retry_note
                + plan_prompt,
                output_type=str,
                native_output=False,
                include_output_context=False,
                include_user_toolsets=False,
                toolsets=[],
            )
            plan_params = plan_request.render(ctx)
            _observe_model_request(observe)
            plan_result = await graph_deps.agent.run(**plan_params)
            run_state.agent_runs.append(plan_result)
            candidate_code: str = plan_result.output
            candidate_code = candidate_code.strip()
            if candidate_code.startswith("```"):
                candidate_code = candidate_code.strip("`")
                if candidate_code.startswith("python"):
                    candidate_code = candidate_code[len("python") :].strip()

            if "complete_task" in candidate_code:
                candidate_code = candidate_code.replace(
                    "complete_task", "complete"
                )

            forbidden = ("await ", "multi_tool_use", "functions.")
            bad_token = next(
                (token for token in forbidden if token in candidate_code),
                None,
            )
            if bad_token:
                plan_error = (
                    f"Plan contains forbidden token: {bad_token!r}. "
                    "Plan must be pure Python code with direct tool calls."
                )
                continue

            plan_code = candidate_code
            plan_error = None
            break

        if plan_code is None:
            raise ValueError(
                f"Failed to generate a valid plan after {max_retries} attempts."
            )

        prompt_blocks: List[str] = []
        code_blocks: List[str] = [plan_code]

        tool_funcs: Dict[str, Callable[..., Any]] = {}

        def _wrap_tool(name: str, tool: Any) -> Callable[..., Any]:
            def _wrapped(*args, **kwargs):
                execution_state.tool_call_log[name] = (
                    execution_state.tool_call_log.get(name, 0) + 1
                )
                execution_state.total_tool_calls += 1
                if (
                    max_steps is not None
                    and execution_state.total_tool_calls > max_steps
                ):
                    raise ValueError(f"Max steps exceeded ({max_steps}).")

                if not isinstance(graph_deps.agent.model, PydanticAIModel):
                    model = graph_deps.agent._get_model(
                        model=graph_deps.agent.model,
                    )
                else:
                    model = graph_deps.agent.model

                ctx = PydanticAIRunContext(
                    deps=graph_deps.deps,
                    model=model,
                    usage=PydanticAIUsage(),
                )
                if getattr(tool, "takes_ctx", False):
                    return tool.function(ctx, *args, **kwargs)
                return tool.function(*args, **kwargs)

            return _wrapped

        for toolset_item in graph_deps.toolsets:
            if isinstance(toolset_item, FunctionToolset):
                for name, tool in toolset_item.tools.items():
                    tool_funcs[name] = _wrap_tool(name, tool)

        def tool_called(name: str) -> bool:
            return execution_state.tool_call_log.get(name, 0) > 0

        def tool_calls(name: str | None = None) -> int | Dict[str, int]:
            if name is None:
                return dict(execution_state.tool_call_log)
            return execution_state.tool_call_log.get(name, 0)

        def prompt(text: str) -> None:
            prompt_blocks.append(text)

        def complete(
            summary: str | None = None,
            *,
            status: Literal["success", "failed", "skipped"] = "success",
            result: Any | None = None,
            evidence: Any | None = None,
            verification: Any | None = None,
        ) -> None:
            _apply_completion(
                state=execution_state,
                summary=summary,
                status=status,
                result=result,
                evidence=evidence,
                verification=verification,
                target=target,
                source=source,
                deps=deps,
                attachments=attachments,
                verifier=verifier,
                final_answer_checks=final_answer_checks,
                error_cls=ValueError,
            )

        external_functions = {
            **tool_funcs,
            "tool_called": tool_called,
            "tool_calls": tool_calls,
            "prompt": prompt,
            "complete": complete,
            "complete_task": complete,
        }

        plan_exec_error: str | None = None
        round_start_calls = execution_state.total_tool_calls
        for _ in range(max_retries):
            try:
                for code in code_blocks:
                    monty = pydantic_monty.Monty(
                        code,
                        inputs=[],
                        external_functions=list(external_functions.keys()),
                        script_name="run_plan.py",
                        type_check=False,
                    )
                    await pydantic_monty.run_monty_async(
                        monty,
                        external_functions=external_functions,
                    )
                if planning_interval is not None:
                    round_calls = (
                        execution_state.total_tool_calls - round_start_calls
                    )
                    if round_calls > planning_interval:
                        raise ValueError(
                            f"Plan exceeded tool-call limit ({planning_interval})."
                        )
                plan_exec_error = None
                break
            except Exception as exc:
                plan_exec_error = str(exc)
                continue

        if plan_exec_error:
            last_plan_error = f"Plan execution failed: {plan_exec_error}"
            if round_idx < plan_rounds - 1:
                continue
            raise ValueError(
                f"Plan execution failed after {plan_rounds} plan rounds: {plan_exec_error}"
            )

        if prompt_blocks:
            prompt_request = SemanticGraphRequestTemplate(
                system_prompt_additions=system_prompt,
                user_prompt_additions="\n\n".join(prompt_blocks),
                output_type=None,
                native_output=False,
                include_output_context=False,
                include_user_toolsets=False,
                toolsets=[],
            )
            prompt_params = prompt_request.render(ctx)
            _observe_model_request(observe)
            prompt_result = await graph_deps.agent.run(**prompt_params)
            run_state.agent_runs.append(prompt_result)

        if execution_state.completion_state is not None:
            break
        last_plan_error = "Plan did not complete the task."

    if require_completion and execution_state.completion_state is None:
        raise ValueError(
            f"Task did not complete after {plan_rounds} plan rounds."
        )


async def _execute_standard_run(
    *,
    ctx: Any,
    graph_deps: SemanticGraphDeps[Deps, Output],
    run_state: SimpleNamespace,
    execution_state: RunExecutionState[Output],
    system_prompt: str,
    toolset: FunctionToolset,
    require_completion: bool,
    max_retries: int,
    max_steps: int | None,
) -> None:
    observe = getattr(graph_deps, "observe", None)
    request = SemanticGraphRequestTemplate(
        system_prompt_additions=system_prompt,
        output_type=None,
        native_output=False,
        include_output_context=False,
        toolsets=[toolset],
    )
    last_error: str | None = None
    max_attempts = max_steps if max_steps is not None else max_retries
    for _ in range(max_attempts):
        if last_error:
            request.user_prompt_additions = (
                f"Previous attempt failed: {last_error}"
            )
        params = request.render(ctx)
        _observe_model_request(observe)
        result = await graph_deps.agent.run(**params)

        run_state.agent_runs.append(result)

        if require_completion and execution_state.completion_state is None:
            last_error = (
                "Task did not complete. You must call `complete_task`."
            )
            continue
        last_error = None
        break

    if last_error:
        raise ValueError(last_error)


async def _execute_stream_run(
    *,
    ctx: Any,
    graph_deps: SemanticGraphDeps[Deps, Output],
    run_state: SimpleNamespace,
    execution_state: RunExecutionState[Output],
    system_prompt: str,
    toolset: FunctionToolset,
    require_completion: bool,
    target: TargetParam[Output],
    task_goal: str | None,
    source: SourceParam | None,
    attachments: AttachmentType | List[AttachmentType] | None,
) -> RunStream[Output]:
    observe = getattr(graph_deps, "observe", None)
    request = SemanticGraphRequestTemplate(
        system_prompt_additions=system_prompt,
        output_type=graph_deps.target_strategy.target
        if graph_deps.target_strategy
        else None,
        native_output=False,
        include_output_context=False,
        toolsets=[toolset],
    )
    params = request.render(ctx)
    handler = (
        getattr(observe, "event_stream_handler", None) if observe else None
    )
    _observe_model_request(observe)
    stream_ctx = await graph_deps.agent.run_stream(
        **params, event_stream_handler=handler
    )
    stream = await stream_ctx.__aenter__()
    builder = OutputBuilder(
        target=graph_deps.target_strategy.target
        if graph_deps.target_strategy
        else target
    )
    stream_wrapper = RunStream(
        _builder=builder,
        _streams=[stream],
        _field_mappings=[
            StreamStreamFieldMapping(
                stream_index=0, fields=None, update_output=True
            )
        ],
        _stream_contexts=[stream_ctx],
        completion=execution_state.completion_state,
        _exclude_none=False,
        verified=execution_state.verified,
        verification_error=execution_state.verification_error,
        task=RunTask(goal=task_goal, target=target, source=source),
        _require_completion=require_completion,
        _execution_state=execution_state,
    )
    return stream_wrapper


async def _run_async(
    task: ContextType | List[ContextType] | None,
    target: TargetParam[Output] | None,
    context: ContextType | List[ContextType] | None,
    *,
    source: SourceParam | None,
    instructions: PydanticAIInstructions | None,
    model: ModelParam,
    model_settings: PydanticAIModelSettings | None,
    attachments: AttachmentType | List[AttachmentType] | None,
    tools: ToolType | List[ToolType] | None,
    deps: Deps | None,
    usage_limits: PydanticAIUsageLimits | None,
    verifier: RunVerifier | None,
    require_completion: bool,
    plan: bool,
    max_retries: int,
    planning_interval: int | None,
    max_steps: int | None,
    final_answer_checks: List[RunCheck] | None,
    observe: bool | "Observer" | None,
    stream: bool = False,
) -> RunResult[Output] | RunStream[Output]:
    if task is None:
        make_target = cast(
            TargetParam[Output], target if target is not None else str
        )
        return await amake(  # type: ignore[no-matching-overload]
            target=make_target,
            context=context,
            instructions=instructions,
            model=model,
            model_settings=model_settings,
            attachments=attachments,
            tools=tools,
            deps=deps,
            usage_limits=usage_limits,
            observe=_normalize_observer(observe),
            stream=stream,
        )

    merged_context = _normalize_context(task=task, context=context)

    _target = target
    _instructions = instructions
    if isinstance(target, Target):
        _target = target.target
        if target.instructions and not _instructions:
            _instructions = target.instructions

    graph_deps = SemanticGraphDeps.prepare(
        model=model,
        model_settings=model_settings,
        context=merged_context,
        instructions=_instructions,  # type: ignore[arg-type]
        tools=tools,
        deps=deps,
        attachments=attachments,
        usage_limits=usage_limits,
        target=_target,
        source=source,
        inject_internal_deps=True,
        observe=_normalize_observer(observe),
    )

    execution_state = RunExecutionState[Output]()

    toolset = FunctionToolset()

    @toolset.tool
    def complete_task(
        summary: str,
        *,
        status: Literal["success", "failed", "skipped"] = "success",
        result: Any | None = None,
        evidence: Any | None = None,
        verification: Any | None = None,
    ) -> str:
        _apply_completion(
            state=execution_state,
            summary=summary,
            status=status,
            result=result,
            evidence=evidence,
            verification=verification,
            target=target,
            source=source,
            deps=deps,
            attachments=attachments,
            verifier=verifier,
            final_answer_checks=final_answer_checks,
            error_cls=ModelRetry,
        )

        return "ok"

    system_prompt = _RUN_SYSTEM_PROMPT
    target_schema = _build_target_schema_text(target)
    if target_schema:
        system_prompt += (
            "\n[EXPECTED RESULT SCHEMA]\n"
            f"{target_schema}\n"
            "[END EXPECTED RESULT SCHEMA]\n"
            "- When calling `complete_task`, ensure `result` matches the schema.\n"
        )
    else:
        system_prompt += (
            "\n- No result schema is required. Use `result=null` unless the task explicitly "
            "requests a specific output.\n"
        )

    run_state = SimpleNamespace(agent_runs=[], output=None)
    ctx = SimpleNamespace(state=run_state, deps=graph_deps)

    if stream and target is None:
        raise ValueError("`stream=True` requires `target` to be provided.")

    if stream and plan:
        raise ValueError("`stream=True` is not supported with `plan=True`.")

    observe = getattr(graph_deps, "observe", None)
    if observe:
        observe.on_operation_start("run")

    try:
        if plan:
            await _execute_plan_based_run(
                ctx=ctx,
                graph_deps=graph_deps,
                run_state=run_state,
                execution_state=execution_state,
                system_prompt=system_prompt,
                target=target,
                source=source,
                deps=deps,
                attachments=attachments,
                verifier=verifier,
                max_retries=max_retries,
                planning_interval=planning_interval,
                max_steps=max_steps,
                final_answer_checks=final_answer_checks,
                require_completion=require_completion,
            )
        else:
            if stream and target is not None:
                task_goal: str | None = None
                if isinstance(task, str):
                    task_goal = task
                return await _execute_stream_run(
                    ctx=ctx,
                    graph_deps=graph_deps,
                    run_state=run_state,
                    execution_state=execution_state,
                    system_prompt=system_prompt,
                    toolset=toolset,
                    require_completion=require_completion,
                    target=target,
                    task_goal=task_goal,
                    source=source,
                    attachments=attachments,
                )
            await _execute_standard_run(
                ctx=ctx,
                graph_deps=graph_deps,
                run_state=run_state,
                execution_state=execution_state,
                system_prompt=system_prompt,
                toolset=toolset,
                require_completion=require_completion,
                max_retries=max_retries,
                max_steps=max_steps,
            )
    finally:
        if observe:
            observe.on_operation_complete("run")

    if require_completion and execution_state.completion_state is None:
        raise ValueError(
            "Task did not complete. The model must call `complete`."
        )

    completion_state = execution_state.completion_state or RunCompletion(
        status="skipped",
        summary="No completion tool call was made.",
    )

    task_goal: str | None = None
    if isinstance(task, str):
        task_goal = task

    ctx_refs = getattr(graph_deps, "_context_refs", None) or []
    if not isinstance(ctx_refs, list):
        ctx_refs = [ctx_refs]
    context_additions = getattr(graph_deps, "_context_additions", []) or []
    for ctx_ref in ctx_refs:
        if ctx_ref is None or not getattr(ctx_ref, "update", True):
            continue
        if context_additions:
            ctx_ref.extend_messages(context_additions)
        ctx_ref.add_assistant_message(
            semantic_for_operation(
                "run",
                summary=completion_state.summary,
                output=execution_state.validated_output,
            )
        )

    return RunResult(
        output=execution_state.validated_output,
        raw=run_state.agent_runs,
        completion=completion_state,
        verified=execution_state.verified,
        verification_error=execution_state.verification_error,
        task=RunTask(goal=task_goal, target=target, source=source),
    )


@overload
async def arun(
    task: ContextType | List[ContextType] | None = ...,
    target: TargetParam[Output] | None = ...,
    context: ContextType | List[ContextType] | None = ...,
    *,
    source: SourceParam | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    attachments: AttachmentType | List[AttachmentType] | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    verifier: RunVerifier | None = ...,
    require_completion: bool = ...,
    plan: bool = ...,
    max_retries: int = ...,
    planning_interval: int | None = ...,
    max_steps: int | None = ...,
    final_answer_checks: List[RunCheck] | None = ...,
    observe: bool | Observer | None = ...,
    stream: Literal[True],
) -> RunStream[Output]: ...


@overload
async def arun(
    task: ContextType | List[ContextType] | None = ...,
    target: TargetParam[Output] | None = ...,
    context: ContextType | List[ContextType] | None = ...,
    *,
    source: SourceParam | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    attachments: AttachmentType | List[AttachmentType] | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    verifier: RunVerifier | None = ...,
    require_completion: bool = ...,
    plan: bool = ...,
    max_retries: int = ...,
    planning_interval: int | None = ...,
    max_steps: int | None = ...,
    final_answer_checks: List[RunCheck] | None = ...,
    observe: bool | Observer | None = ...,
    stream: Literal[False] = False,
) -> RunResult[Output]: ...


async def arun(
    task: ContextType | List[ContextType] | None = None,
    target: TargetParam[Output] | None = None,
    context: ContextType | List[ContextType] | None = None,
    *,
    source: SourceParam | None = None,
    instructions: PydanticAIInstructions | None = None,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: PydanticAIModelSettings | None = None,
    attachments: AttachmentType | List[AttachmentType] | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    usage_limits: PydanticAIUsageLimits | None = None,
    verifier: RunVerifier | None = None,
    require_completion: bool = True,
    plan: bool = False,
    max_retries: int = 2,
    planning_interval: int | None = None,
    max_steps: int | None = None,
    final_answer_checks: List[RunCheck] | None = None,
    observe: bool | Observer | None = None,
    stream: bool = False,
) -> RunResult[Output] | RunStream[Output]:
    """Asynchronously run a task with tool-verified completion.

    `run()` is goal-centric: completion is signaled by a tool call, not by a
    typed output. If `target` is provided, the completion tool must include
    a result that validates against the target schema.
    """
    return await _run_async(
        task=task,
        target=target,
        context=context,
        source=source,
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        attachments=attachments,
        tools=tools,
        deps=deps,
        usage_limits=usage_limits,
        verifier=verifier,
        require_completion=require_completion,
        plan=plan,
        max_retries=max_retries,
        planning_interval=planning_interval,
        max_steps=max_steps,
        final_answer_checks=final_answer_checks,
        observe=observe,
        stream=stream,
    )


@overload
def run(
    task: ContextType | List[ContextType] | None = ...,
    target: TargetParam[Output] | None = ...,
    context: ContextType | List[ContextType] | None = ...,
    *,
    source: SourceParam | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    attachments: AttachmentType | List[AttachmentType] | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    verifier: RunVerifier | None = ...,
    require_completion: bool = ...,
    plan: bool = ...,
    max_retries: int = ...,
    planning_interval: int | None = ...,
    max_steps: int | None = ...,
    final_answer_checks: List[RunCheck] | None = ...,
    observe: bool | Observer | None = ...,
    stream: Literal[True],
) -> RunStream[Output]: ...


@overload
def run(
    task: ContextType | List[ContextType] | None = ...,
    target: TargetParam[Output] | None = ...,
    context: ContextType | List[ContextType] | None = ...,
    *,
    source: SourceParam | None = ...,
    instructions: PydanticAIInstructions | None = ...,
    model: ModelParam = ...,
    model_settings: PydanticAIModelSettings | None = ...,
    attachments: AttachmentType | List[AttachmentType] | None = ...,
    tools: ToolType | List[ToolType] | None = ...,
    deps: Deps | None = ...,
    usage_limits: PydanticAIUsageLimits | None = ...,
    verifier: RunVerifier | None = ...,
    require_completion: bool = ...,
    plan: bool = ...,
    max_retries: int = ...,
    planning_interval: int | None = ...,
    max_steps: int | None = ...,
    final_answer_checks: List[RunCheck] | None = ...,
    observe: bool | Observer | None = ...,
    stream: Literal[False] = False,
) -> RunResult[Output]: ...


def run(
    task: ContextType | List[ContextType] | None = None,
    target: TargetParam[Output] | None = None,
    context: ContextType | List[ContextType] | None = None,
    *,
    source: SourceParam | None = None,
    instructions: PydanticAIInstructions | None = None,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: PydanticAIModelSettings | None = None,
    attachments: AttachmentType | List[AttachmentType] | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    usage_limits: PydanticAIUsageLimits | None = None,
    verifier: RunVerifier | None = None,
    require_completion: bool = True,
    plan: bool = False,
    max_retries: int = 2,
    planning_interval: int | None = None,
    max_steps: int | None = None,
    final_answer_checks: List[RunCheck] | None = None,
    observe: bool | Observer | None = None,
    stream: bool = False,
) -> RunResult[Output] | RunStream[Output]:
    """Run a task with tool-verified completion."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "Cannot call `run()` inside an async context. Use `await arun()`."
        )

    return asyncio.run(
        _run_async(
            task=task,
            target=target,
            context=context,
            source=source,
            instructions=instructions,
            model=model,
            model_settings=model_settings,
            attachments=attachments,
            tools=tools,
            deps=deps,
            usage_limits=usage_limits,
            verifier=verifier,
            require_completion=require_completion,
            plan=plan,
            max_retries=max_retries,
            planning_interval=planning_interval,
            max_steps=max_steps,
            final_answer_checks=final_answer_checks,
            observe=observe,
            stream=stream,
        )
    )
