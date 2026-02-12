"""zyx.operations.validate"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, List, TypeVar

from pydantic import BaseModel, Field

from .._aliases import (
    PydanticAIAgentResult,
    PydanticAIInstructions,
    PydanticAIModelSettings,
    PydanticAIUsageLimits,
)
from .._graph import (
    SemanticGraph,
    SemanticGraphDeps,
    SemanticGraphState,
    SemanticGenerateNode,
    SemanticGraphRequestTemplate,
)
from .._types import (
    ModelParam,
    SourceParam,
    TargetParam,
    ContextType,
    ToolType,
    AttachmentType,
)
from ..result import Result
from .parse import aparse, parse

__all__ = (
    "avalidate",
    "validate",
    "ConstraintViolation",
    "ConstraintViolations",
    "ValidationResult",
)

Deps = TypeVar("Deps")
Output = TypeVar("Output")


class ConstraintViolation(BaseModel):
    """A single constraint that was violated during validation."""

    constraint: str = Field(
        description="The exact constraint text that was violated."
    )
    reason: str = Field(
        description="Human-readable explanation of why the value violates this constraint."
    )


class ConstraintViolations(BaseModel):
    """Aggregated result of constraint validation.

    When every constraint is satisfied the ``violations`` list is empty.
    """

    violations: List[ConstraintViolation] = Field(
        default_factory=list,
        description=(
            "List of constraints that were violated. "
            "Empty when the value satisfies all constraints."
        ),
    )


@dataclass
class ValidationResult(Generic[Output]):
    """Result of validate when ``raise_on_error=False``.

    Contains the parsed output and constraint violations (empty when valid).
    """

    output: Output
    """The parsed output from the parse step."""

    violations: ConstraintViolations
    """Constraint validation result; ``violations.violations`` is empty when valid."""

    raw: List[PydanticAIAgentResult[Any]]
    """Underlying agent run results (parse + validate)."""


_VALIDATE_SYSTEM_PROMPT = (
    "\n[INSTRUCTION]\n"
    "- You are a constraint validator. Your ONLY task is to check whether "
    "the given value satisfies a set of constraints.\n"
    "- Evaluate every constraint carefully.\n"
    "- Return a ConstraintViolations object.\n"
    "- If ALL constraints are satisfied, return ConstraintViolations with an "
    "empty violations list.\n"
    "- If ANY constraint is violated, include an entry in the violations list "
    "with the exact constraint text and a concise reason.\n"
)


def _build_constraint_list(constraints: List[str]) -> str:
    """Format a numbered constraint list for inclusion in a system prompt."""
    lines = ["\n[CONSTRAINTS]"]
    for i, c in enumerate(constraints, 1):
        lines.append(f"  {i}. {c}")
    return "\n".join(lines)


def _build_validate_prompt(constraints: List[str], parsed_value: Any) -> str:
    """Build validation prompt that includes the parsed value and constraints."""
    from .._processing._toon import object_as_toon_text

    constraint_text = _build_constraint_list(constraints)
    parsed_repr = object_as_toon_text(parsed_value)
    return (
        _VALIDATE_SYSTEM_PROMPT + constraint_text + "\n"
        f"\n[PARSED VALUE]\n"
        f"The value to validate:\n{parsed_repr}\n"
        f"\nValidate this parsed value against ALL constraints listed above.\n"
    )


def _raise_on_violations(
    violations: ConstraintViolations, parsed: Any
) -> None:
    """Raise ``AssertionError`` if *violations* contains any entries.

    Includes the parsed value in the error message for context.
    """
    if violations.violations:
        from .._processing._toon import object_as_toon_text

        formatted = "\n".join(
            f"  - {v.constraint}: {v.reason}" for v in violations.violations
        )
        parsed_repr = object_as_toon_text(parsed)
        raise AssertionError(
            f"Constraint validation failed for parsed value:\n{parsed_repr}\n\n"
            f"Violations:\n{formatted}"
        )


async def avalidate(
    source: SourceParam,
    target: TargetParam[Output] = str,  # type: ignore[assignment]
    context: ContextType | List[ContextType] | None = None,
    *,
    constraints: List[str] | None = None,
    raise_on_error: bool = True,
    confidence: bool = False,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: PydanticAIModelSettings | None = None,
    attachments: AttachmentType | List[AttachmentType] | None = None,
    instructions: PydanticAIInstructions | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    usage_limits: PydanticAIUsageLimits | None = None,
) -> Result[Output] | ValidationResult[Output]:
    """Asynchronously parse a source into a target type, then validate the result against constraints.

    Args:
        source (SourceParam): The value to parse and validate.
        target (TargetParam[Output]): The type or schema to parse into. Defaults to str.
        context (ContextType | List[ContextType] | None): Optional context or conversation history. Defaults to None.
        constraints (List[str] | None): List of constraint strings to validate against. Defaults to None.
        raise_on_error (bool): If True (default), raise AssertionError if validation fails.
            If False, return a ValidationResult with any violations. Defaults to True.
        confidence (bool): If True, includes confidence scoring. Defaults to False.
        model (ModelParam): The model to use for parsing/validation. Defaults to "openai:gpt-4o-mini".
        model_settings (PydanticAIModelSettings | None): Model settings to use for the operation. Defaults to None.
        instructions (PydanticAIInstructions | None): Additional instructions for the model. Defaults to None.
        attachments (AttachmentType | List[AttachmentType] | None): Attachments to provide to the model. Defaults to None.
        tools (ToolType | List[ToolType] | None): Tools available to the model. Defaults to None.
        deps (Deps | None): Optional RunContext dependencies. Defaults to None.
        usage_limits (PydanticAIUsageLimits | None): Usage limits for the model call. Defaults to None.

    Returns:
        Result[Output] | ValidationResult[Output]: Result[Output] if raise_on_error is True and validation passes.
            ValidationResult[Output] if raise_on_error is False (always includes violations).
    """
    from ..targets import Target

    _constraints: List[str] = constraints or []
    if isinstance(target, Target) and getattr(target, "constraints", None):
        _constraints = target.constraints or _constraints  # type: ignore[assignment]

    parse_result = await aparse(
        source=source,
        target=target,
        context=context,
        confidence=confidence,
        model=model,
        model_settings=model_settings,
        instructions=instructions,
        tools=tools,
        deps=deps,
        attachments=attachments,
        usage_limits=usage_limits,
        stream=False,
    )
    parsed_value = parse_result.output

    if _constraints:
        validate_deps = SemanticGraphDeps.prepare(
            target=ConstraintViolations,
            source=None,
            model=model,
            model_settings=model_settings,
            context=None,
            instructions=_build_validate_prompt(_constraints, parsed_value),
            tools=None,
            deps=deps,
            attachments=attachments,
            usage_limits=usage_limits,
        )
        validate_state = SemanticGraphState.prepare(deps=validate_deps)
        validate_request = SemanticGraphRequestTemplate(
            include_source_context=False,
        )
        validate_graph = SemanticGraph(
            nodes=[SemanticGenerateNode],
            start=SemanticGenerateNode(
                request=validate_request,
                update_output=False,
            ),
            state=validate_state,
            deps=validate_deps,
        )
        validate_result = await validate_graph.run()
        violations = validate_result.output
        if (
            isinstance(violations, ConstraintViolations)
            and violations.violations
        ):
            if raise_on_error:
                _raise_on_violations(violations, parsed_value)
            return ValidationResult(
                output=parsed_value,
                violations=violations,
                raw=parse_result.raw + validate_result.raw,
            )
        if raise_on_error:
            return parse_result
        return ValidationResult(
            output=parsed_value,
            violations=violations,
            raw=parse_result.raw + validate_result.raw,
        )

    else:
        return parse_result


def validate(
    source: SourceParam,
    target: TargetParam[Output] = str,  # type: ignore[assignment]
    context: ContextType | List[ContextType] | None = None,
    *,
    constraints: List[str] | None = None,
    raise_on_error: bool = True,
    confidence: bool = False,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: PydanticAIModelSettings | None = None,
    attachments: AttachmentType | List[AttachmentType] | None = None,
    instructions: PydanticAIInstructions | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    usage_limits: PydanticAIUsageLimits | None = None,
) -> Result[Output] | ValidationResult[Output]:
    """Synchronously parse a source into a target type, then validate the result against constraints.

    Args:
        source (SourceParam): The value to parse and validate.
        target (TargetParam[Output]): The type or schema to parse into. Defaults to str.
        context (ContextType | List[ContextType] | None): Optional context or conversation history. Defaults to None.
        constraints (List[str] | None): List of constraint strings to validate against. Defaults to None.
        raise_on_error (bool): If True (default), raise AssertionError if validation fails.
            If False, return a ValidationResult with any violations. Defaults to True.
        confidence (bool): If True, includes confidence scoring. Defaults to False.
        model (ModelParam): The model to use for parsing/validation. Defaults to "openai:gpt-4o-mini".
        model_settings (PydanticAIModelSettings | None): Model settings to use for the operation. Defaults to None.
        instructions (PydanticAIInstructions | None): Additional instructions for the model. Defaults to None.
        attachments (AttachmentType | List[AttachmentType] | None): Attachments to provide to the model. Defaults to None.
        tools (ToolType | List[ToolType] | None): Tools available to the model. Defaults to None.
        deps (Deps | None): Optional RunContext dependencies. Defaults to None.
        usage_limits (PydanticAIUsageLimits | None): Usage limits for the model call. Defaults to None.

    Returns:
        Result[Output] | ValidationResult[Output]: Result[Output] if raise_on_error is True and validation passes.
            ValidationResult[Output] if raise_on_error is False (always includes violations).
    """
    from ..targets import Target

    _constraints: List[str] = constraints or []
    if isinstance(target, Target) and getattr(target, "constraints", None):
        _constraints = target.constraints or _constraints  # type: ignore[assignment]

    parse_result = parse(
        source=source,
        target=target,
        context=context,
        confidence=confidence,
        model=model,
        model_settings=model_settings,
        attachments=attachments,
        instructions=instructions,
        tools=tools,
        deps=deps,
        usage_limits=usage_limits,
        stream=False,
    )
    parsed_value = parse_result.output

    if _constraints:
        validate_deps = SemanticGraphDeps.prepare(
            target=ConstraintViolations,
            source=None,
            model=model,
            model_settings=model_settings,
            context=None,
            instructions=_build_validate_prompt(_constraints, parsed_value),
            tools=None,
            deps=deps,
            attachments=attachments,
            usage_limits=usage_limits,
        )
        validate_state = SemanticGraphState.prepare(deps=validate_deps)
        validate_request = SemanticGraphRequestTemplate(
            include_source_context=False,
        )
        validate_graph = SemanticGraph(
            nodes=[SemanticGenerateNode],
            start=SemanticGenerateNode(
                request=validate_request,
                update_output=False,
            ),
            state=validate_state,
            deps=validate_deps,
        )
        validate_result = validate_graph.run_sync()
        violations = validate_result.output
        if (
            isinstance(violations, ConstraintViolations)
            and violations.violations
        ):
            if raise_on_error:
                _raise_on_violations(violations, parsed_value)
            return ValidationResult(
                output=parsed_value,
                violations=violations,
                raw=parse_result.raw + validate_result.raw,
            )
        if raise_on_error:
            return parse_result
        return ValidationResult(
            output=parsed_value,
            violations=violations,
            raw=parse_result.raw + validate_result.raw,
        )

    else:
        return parse_result
