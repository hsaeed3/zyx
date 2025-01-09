"""
Tool calling helper tests
"""

import pytest
from rich import print
from pydantic import BaseModel
from zyx.utils import tool_calling as tool_calling_helpers

module_tag = "[bold italic sky_blue3]zyx.utils.tool_calling[/bold italic sky_blue3]"


# ===================================================================
# Tool Conversion
# ===================================================================


def test_utils_tool_calling_convert_to_openai_tool_from_function() -> None:
    """Test converting a Python function to an OpenAI tool."""

    def sample_function(x: int) -> int:
        return x + 1

    tool = tool_calling_helpers.convert_to_openai_tool(sample_function)
    print(f"{module_tag} - [bold green]Tool from Function[/bold green]")
    print(tool)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "sample_function"
    assert tool["function"]["parameters"]["type"] == "object"
    assert "properties" in tool["function"]["parameters"]
    assert "x" in tool["function"]["parameters"]["properties"]
    assert tool["function"]["parameters"]["properties"]["x"]["type"] == "integer"


def test_utils_tool_calling_convert_to_openai_tool_from_pydantic() -> None:
    """Test converting a Pydantic model to an OpenAI tool."""

    class SampleModel(BaseModel):
        x: int

    model_instance = SampleModel(x=1)
    tool = tool_calling_helpers.convert_to_openai_tool(model_instance)
    print(f"{module_tag} - [bold green]Tool from Pydantic Model[/bold green]")
    print(tool)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "SampleModel"
    assert tool["function"]["parameters"]["type"] == "object"
    assert "properties" in tool["function"]["parameters"]
    assert "x" in tool["function"]["parameters"]["properties"]
    assert tool["function"]["parameters"]["properties"]["x"]["type"] == "integer"


def test_utils_tool_calling_convert_to_openai_tool_from_dict() -> None:
    """Test converting an OpenAI tool dictionary."""
    tool_dict = {"type": "function", "function": {"name": "calculator", "parameters": {"x": 1, "y": 2}}}
    tool = tool_calling_helpers.convert_to_openai_tool(tool_dict)
    print(f"{module_tag} - [bold green]Tool from Dictionary[/bold green]")
    print(tool)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "calculator"
    assert tool["function"]["parameters"]["x"] == 1
    assert tool["function"]["parameters"]["y"] == 2


def test_utils_tool_calling_convert_to_openai_tool_invalid_dict() -> None:
    """Test converting an invalid tool dictionary raises an exception."""
    invalid_tool_dict = {"name": "invalid_tool"}
    with pytest.raises(Exception):
        tool_calling_helpers.convert_to_openai_tool(invalid_tool_dict)


if __name__ == "__main__":
    test_utils_tool_calling_convert_to_openai_tool_from_function()
    test_utils_tool_calling_convert_to_openai_tool_from_pydantic()
    test_utils_tool_calling_convert_to_openai_tool_from_dict()
    test_utils_tool_calling_convert_to_openai_tool_invalid_dict()
