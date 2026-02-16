"""tests.test_utils_semantic"""

from dataclasses import dataclass

from zyx._utils._semantic import (
    semantic_from_output,
    semantic_for_edit,
    semantic_for_run,
    semantic_for_operation,
)


@dataclass
class Person:
    name: str
    age: int


def test_semantic_from_output_none():
    assert semantic_from_output(None) == "Completed."


def test_semantic_for_edit_tracks_changed_fields():
    original = {"a": 1, "b": 2}
    updated = {"a": 1, "b": 3}
    result = semantic_for_edit(original, updated)
    assert "Edited fields: b" in result


def test_semantic_for_run_summary_and_output():
    result = semantic_for_run("ok", {"value": 1})
    assert "Task completed: ok" in result
    assert "Result:" in result


def test_semantic_for_operation_switching():
    edit_result = semantic_for_operation(
        "edit", original=Person("A", 1), updated=Person("A", 2)
    )
    assert "Edited" in edit_result

    run_result = semantic_for_operation("run", summary="done", output=1)
    assert "Task completed: done" in run_result

    default_result = semantic_for_operation("make", output="hi")
    assert "hi" in default_result
