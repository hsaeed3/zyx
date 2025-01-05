"""
Tool calling helper tests
"""

import pytest
from rich import print
from pydantic import BaseModel
from zyx.core.helpers import tool_calling as tool_calling_helpers

module_tag = "[bold italic sky_blue3]zyx.helpers.tool_calling[/bold italic sky_blue3]"


# ===================================================================
# Tool Conversion
# ===================================================================


def test_core_helpers_tool_calling_convert_to_openai_tool():
    # Test with a Python function
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

    # Test with a Pydantic model
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

    # Test with an OpenAI tool dictionary
    tool_dict = {"type": "function", "function": {"name": "calculator", "parameters": {"x": 1, "y": 2}}}
    tool = tool_calling_helpers.convert_to_openai_tool(tool_dict)
    print(f"{module_tag} - [bold green]Tool from Dictionary[/bold green]")
    print(tool)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "calculator"
    assert tool["function"]["parameters"]["x"] == 1
    assert tool["function"]["parameters"]["y"] == 2

    # Test with invalid tool dictionary
    invalid_tool_dict = {"name": "invalid_tool"}
    with pytest.raises(Exception):
        tool_calling_helpers.convert_to_openai_tool(invalid_tool_dict)


if __name__ == "__main__":
    test_core_helpers_tool_calling_convert_to_openai_tool()
