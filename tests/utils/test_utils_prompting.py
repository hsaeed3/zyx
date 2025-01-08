# zyx.utils.prompting tests

from typing import Any
import logging
import pytest
from rich import print
from pydantic import BaseModel
from zyx.utils import prompting

module_tag = "[bold italic sky_blue3]zyx.utils.prompting[/bold italic sky_blue3]"
logger = logging.getLogger("zyx")


# ===================================================================
# Context String Creation
# ===================================================================


def test_utils_prompting_convert_object_to_prompt_context_string() -> None:
    """Test converting a simple string to context."""

    context = "Hello, world!"
    result = prompting.convert_object_to_prompt_context(context)
    assert isinstance(result, str)
    print(f"{module_tag} - [bold green]String Context[/bold green]")
    print(result)

    assert result == "Hello, world!"


def test_utils_prompting_convert_object_to_prompt_context_dict() -> None:
    """Test converting a dictionary to context."""

    

    context = {"key": "value", "number": 42}
    result = prompting.convert_object_to_prompt_context(context)
    assert isinstance(result, str)
    print(f"{module_tag} - [bold green]Dict Context[/bold green]")
    print(result)

    assert '"key": "value"' in result
    assert '"number": 42' in result


def test_utils_prompting_convert_object_to_prompt_context_pydantic() -> None:
    """Test converting a Pydantic model to context."""

    

    class TestModel(BaseModel):
        name: str
        age: int
        description: str | None = None

    model = TestModel(name="Test", age=25)
    result = prompting.convert_object_to_prompt_context(model, output_format = "markdown")
    assert isinstance(result, str)
    print(f"{module_tag} - [bold green]Pydantic Context[/bold green]")
    print(result)
    

def test_utils_prompting_convert_object_to_prompt_context_markdown() -> None:
    """Test converting objects with markdown formatting."""

    

    # Test string markdown
    string_result = prompting.convert_object_to_prompt_context("Hello", output_format="markdown")
    assert isinstance(string_result, str)
    print(f"{module_tag} - [bold green]String Markdown[/bold green]")
    print(string_result)
    assert "```text" in string_result
    assert "Hello" in string_result

    # Test dict markdown
    dict_result = prompting.convert_object_to_prompt_context({"key": "value"}, output_format="markdown")
    assert isinstance(dict_result, str)
    print(f"{module_tag} - [bold green]Dict Markdown[/bold green]")
    print(dict_result)
    assert "# dict Object" in dict_result
    assert "```json" in dict_result


def test_utils_prompting_convert_object_to_prompt_context_invalid() -> None:
    """Test handling of invalid inputs."""

    

    class BadObject:
        def __repr__(self) -> str:
            raise Exception("Bad object")

    with pytest.raises(Exception):
        prompting.convert_object_to_prompt_context(BadObject())


if __name__ == "__main__":
    test_utils_prompting_convert_object_to_prompt_context_string()
    test_utils_prompting_convert_object_to_prompt_context_dict()
    test_utils_prompting_convert_object_to_prompt_context_pydantic()
    test_utils_prompting_convert_object_to_prompt_context_markdown()
    test_utils_prompting_convert_object_to_prompt_context_invalid()

