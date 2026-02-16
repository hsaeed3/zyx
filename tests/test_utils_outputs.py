"""tests.test_utils_outputs"""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from zyx._utils._outputs import OutputBuilder


class PersonModel(BaseModel):
    name: str
    age: int


@dataclass
class Person:
    name: str
    age: int


class DummyResult:
    def __init__(self, output):
        self.output = output


def test_output_builder_partial_type_optional():
    builder = OutputBuilder(target=PersonModel)
    partial_type = builder.partial_type
    for field in partial_type.model_fields.values():
        assert not field.is_required()


def test_output_builder_update_and_missing_fields():
    builder = OutputBuilder(target=PersonModel)
    builder.update({"name": "A"})
    assert builder.missing_fields == ["age"]
    assert not builder.is_complete

    builder.update({"age": 2})
    assert builder.is_complete
    result = builder.finalize()
    assert isinstance(result, PersonModel)
    assert result.name == "A"
    assert result.age == 2


def test_output_builder_field_level_update_from_result():
    builder = OutputBuilder(target=PersonModel)
    result = DummyResult(output=5)
    builder.update_from_pydantic_ai_result(result, fields="age") # type: ignore[arg-type]
    assert builder.partial is not None
    assert builder.partial.age == 5
    assert builder.missing_fields == ["name"]


def test_output_builder_finalize_dict_target():
    target = {"name": str, "age": int}
    builder = OutputBuilder(target=target)
    builder.update({"name": "A", "age": 3})
    finalized = builder.finalize()
    assert isinstance(finalized, dict)
    assert finalized == {"name": "A", "age": 3}
