"""
Prompting helper tests
"""

import pytest
from rich import print
from pydantic import BaseModel
from zyx.core.helpers import context as context_helpers

module_tag = "[bold italic sky_blue3]zyx.helpers.prompting[/bold italic sky_blue3]"


# ===================================================================
# Context String Creation
# ===================================================================


def test_core_helpers_context_convert_context_to_string():
    # Test with string
    context_str = context_helpers.convert_context_to_string("Hello")
    print(context_str)
    assert context_str == "Hello"

    # Test with dict
    context_dict = {"key": "value"}
    context_str = context_helpers.convert_context_to_string(context_dict)
    print(context_str)
    assert context_str == '{"key": "value"}'

    # Test with pydantic model class
    class TestModel(BaseModel):
        name: str
        age: int

    context_str = context_helpers.convert_context_to_string(TestModel)
    print(context_str)
    assert "properties" in context_str
    assert "name" in context_str
    assert "age" in context_str

    # Test with pydantic model instance
    model = TestModel(name="test", age=25)
    context_str = context_helpers.convert_context_to_string(model)
    print(context_str)
    assert '"name":"test"' in context_str
    assert '"age":25' in context_str


if __name__ == "__main__":
    test_core_helpers_context_convert_context_to_string()
