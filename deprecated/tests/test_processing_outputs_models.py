"""tests.test_processing_outputs_models"""

import pytest
from pydantic import BaseModel

from zyx._processing._outputs import (
    partial_output_model,
    selection_output_model,
    sparse_output_model,
    split_output_model_by_fields,
    split_output_model,
)


class User(BaseModel):
    name: str
    age: int
    email: str | None = None


def test_partial_output_model_optional_fields():
    Partial = partial_output_model(User)
    for field in Partial.model_fields.values():
        assert not field.is_required()


def test_selection_output_model_literal():
    Selection = selection_output_model(["a", "b"], literal=True)
    Selection.model_validate({"index": "a"})
    with pytest.raises(Exception):
        Selection.model_validate({"index": "c"})


def test_selection_output_model_multi_label_bounds():
    Selection = selection_output_model(["x", "y"], multi_label=True)
    Selection.model_validate({"indices": [0, 1]})
    with pytest.raises(Exception):
        Selection.model_validate({"indices": [2]})


def test_sparse_output_model_changes():
    Updates = sparse_output_model(User)
    parsed = Updates.model_validate(
        {"changes": [{"field": "name", "value": "A"}]}
    )
    assert parsed.changes[0].field == "name"  # type: ignore[attr-defined]
    assert parsed.changes[0].value == "A"  # type: ignore[attr-defined]


def test_split_output_model_by_fields():
    split = split_output_model_by_fields(User)
    assert set(split.keys()) == {"name", "age", "email"}
    NameModel = split["name"]
    assert "content" in NameModel.model_fields


def test_split_output_model_partial():
    Split = split_output_model(User, ["name", "email"], partial=True)
    assert set(Split.model_fields.keys()) == {"name", "email"}
    for field in Split.model_fields.values():
        assert not field.is_required()
