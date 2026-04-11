"""tests.test_processing_outputs"""

import pytest

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel
from zyx._processing._outputs import normalize_output_target


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Person:
    name: str
    age: int


class User(BaseModel):
    username: str
    email: str


def test_primitives():
    assert normalize_output_target(str) is str
    assert normalize_output_target(int) is int
    assert normalize_output_target(float) is float
    print("✓ Primitives work")


def test_generic_list():
    result = normalize_output_target(list[str])
    print(f"✓ list[str] works: {result}")


def test_enum():
    assert normalize_output_target(Color) is Color
    print("✓ Enum works")


def test_literal():
    lit_type = Literal["a", "b", "c"]
    result = normalize_output_target(lit_type)
    print(f"✓ Literal works: {result}")


def test_basemodel():
    assert normalize_output_target(User) is User
    print("✓ BaseModel works")


def test_dataclass_type():
    result = normalize_output_target(Person)
    print(f"✓ Dataclass type works: {result}")
    assert issubclass(result, BaseModel)


def test_dataclass_instance():
    person_instance = Person(name="John", age=30)
    result = normalize_output_target(person_instance)
    print(f"✓ Dataclass instance works: {result}")
    assert issubclass(result, BaseModel)


def test_dict_schema():
    schema = {"name": str, "age": int, "active": True}
    result = normalize_output_target(schema)
    print(f"✓ Dict schema works: {result}")
    assert issubclass(result, BaseModel)
    print(f"  Fields: {result.model_fields.keys()}")


def test_caching():
    # Call twice to ensure caching works
    result1 = normalize_output_target(str)
    result2 = normalize_output_target(str)
    assert result1 is result2
    print("✓ Caching works for primitives")

    result1 = normalize_output_target(Person)
    result2 = normalize_output_target(Person)
    assert result1 is result2
    print("✓ Caching works for dataclasses")


if __name__ == "__main__":
    pytest.main(["-s", __file__])
