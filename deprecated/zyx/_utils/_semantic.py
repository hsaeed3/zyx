"""zyx.utils.semantic"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

from pydantic import BaseModel

from .._processing._toon import object_as_text

__all__ = (
    "semantic_from_output",
    "semantic_for_edit",
    "semantic_for_run",
    "semantic_for_operation",
)


def _as_mapping(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return value
    return None


def semantic_from_output(output: Any) -> str:
    """Render an output value for semantic context updates."""
    if output is None:
        return "Completed."
    return object_as_text(output)


def semantic_for_edit(original: Any, updated: Any) -> str:
    """Render a concise semantic summary for edit operations."""
    original_map = _as_mapping(original)
    updated_map = _as_mapping(updated)

    if original_map is not None and updated_map is not None:
        changed: List[str] = []
        for key, new_value in updated_map.items():
            old_value = original_map.get(key, None)
            if old_value != new_value:
                changed.append(key)

        if changed:
            changed_list = ", ".join(changed)
            return (
                f"Edited fields: {changed_list}. "
                f"Result: {object_as_text(updated)}"
            )

    return f"Edited target. Result: {object_as_text(updated)}"


def semantic_for_run(summary: str | None, output: Any) -> str:
    if summary:
        if output is not None:
            return (
                f"Task completed: {summary}. Result: {object_as_text(output)}"
            )
        return f"Task completed: {summary}."
    if output is not None:
        return f"Task completed. Result: {object_as_text(output)}"
    return "Task completed."


def semantic_for_operation(
    operation: str,
    *,
    output: Any | None = None,
    original: Any | None = None,
    updated: Any | None = None,
    summary: str | None = None,
) -> str:
    op = operation.lower()
    if op == "edit":
        return semantic_for_edit(original, updated)
    if op == "run":
        return semantic_for_run(summary, output)
    return semantic_from_output(output)
