"""tests.test_processing_toon"""

from pydantic import BaseModel

from zyx._processing._toon import object_as_text


class User(BaseModel):
    name: str
    age: int


def test_object_as_text_simple_type():
    assert object_as_text(str) == "string"


def test_object_as_text_model_type_contains_fields():
    result = object_as_text(User)
    assert "name" in result
    assert "age" in result


def test_object_as_text_model_value_contains_fields():
    result = object_as_text(User(name="A", age=1))
    assert "name" in result
    assert "age" in result
