"""Tests for zyx.utils.prompting module."""

from typing import Any, Dict, List, Type, TYPE_CHECKING
import logging
import pytest
from rich import print
from pydantic import BaseModel
from zyx.utils import prompting

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

module_tag = "[bold italic sky_blue3]zyx.utils.prompting[/bold italic sky_blue3]"
logger = logging.getLogger("zyx")


def test_utils_prompting_convert_object_to_prompt_context_string() -> None:
    """
    Test converting a simple string to context.

    Verifies that a string input is handled correctly and returned as-is when
    no markdown formatting is requested.
    """
    context = "Hello, world!"
    result = prompting.convert_object_to_prompt_context(context)
    assert isinstance(result, str)
    print(f"{module_tag} - [bold green]String Context[/bold green]")
    print(result)
    assert result == "Hello, world!"


def test_utils_prompting_convert_object_to_prompt_context_dict() -> None:
    """
    Test converting a dictionary to context.

    Verifies that a dictionary is properly converted to a JSON string representation.
    """
    context: Dict[str, Any] = {"key": "value", "number": 42}
    result = prompting.convert_object_to_prompt_context(context)
    assert isinstance(result, str)
    print(f"{module_tag} - [bold green]Dict Context[/bold green]")
    print(result)
    assert '"key": "value"' in result
    assert '"number": 42' in result


def test_utils_prompting_convert_object_to_prompt_context_pydantic() -> None:
    """
    Test converting a Pydantic model to context.

    Verifies that a Pydantic model instance is properly converted to a markdown
    formatted string with model information.
    """

    class TestModel(BaseModel):
        """Test model for context conversion."""

        name: str
        age: int
        description: str | None = None

    model = TestModel(name="Test", age=25)
    result = prompting.convert_object_to_prompt_context(model, output_format="markdown")
    assert isinstance(result, str)
    print(f"{module_tag} - [bold green]Pydantic Context[/bold green]")
    print(result)
    assert "TestModel Object" in result
    assert "name" in result
    assert "age" in result


def test_utils_prompting_convert_object_to_prompt_context_markdown() -> None:
    """
    Test converting objects with markdown formatting.

    Verifies that different object types are properly formatted with markdown syntax.
    """
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
    assert "dict Object" in dict_result
    assert "```json" in dict_result


def test_utils_prompting_convert_object_to_prompt_context_invalid() -> None:
    """
    Test handling of invalid inputs.

    Verifies that appropriate exceptions are raised for objects that cannot be converted.
    """

    class BadObject:
        def __repr__(self) -> str:
            raise Exception("Bad object")

    with pytest.raises(Exception):
        prompting.convert_object_to_prompt_context(BadObject())


def test_utils_prompting_construct_model_prompt_class() -> None:
    """
    Test constructing prompt from Pydantic model class.

    Verifies that model class schema information is properly formatted in the prompt.
    """

    class TestModel(BaseModel):
        """A test model."""

        name: str
        age: int
        description: str | None = None

    result = prompting.construct_model_prompt(TestModel)
    assert isinstance(result, str)
    print(f"{module_tag} - [bold green]Model Class Prompt[/bold green]")
    print(result)
    assert "TestModel Schema" in result
    assert "Fields" in result
    assert "name" in result
    assert "age" in result
    assert "description" in result


def test_utils_prompting_construct_model_prompt_instance() -> None:
    """Test constructing prompt from Pydantic model instance."""

    class TestModel(BaseModel):
        """A test model."""

        name: str
        age: int
        description: str | None = None

    model = TestModel(name="Test", age=25)
    result = prompting.construct_model_prompt(model)
    assert isinstance(result, str)
    print(f"{module_tag} - [bold green]Model Instance Prompt[/bold green]")
    print(result)
    assert "TestModel Instance" in result
    assert "Fields and Values" in result
    assert "Test" in result
    assert "25" in result


def test_utils_prompting_construct_system_prompt_dict() -> None:
    """Test constructing system prompt from dictionary."""
    target = {"section1": "content1", "section2": "content2"}
    result = prompting.construct_system_prompt(target)
    assert isinstance(result, prompting.Prompt)
    sections = result.sections
    assert len(sections) == 2
    assert sections[0].title == "section1"
    assert sections[0].content == "content1"
    assert sections[1].title == "section2"
    assert sections[1].content == "content2"


def test_utils_prompting_construct_system_prompt_model() -> None:
    """Test constructing system prompt from Pydantic model."""

    class TestModel(BaseModel):
        """A test model."""

        name: str
        age: int

    model = TestModel(name="Test", age=25)
    result = prompting.construct_system_prompt(model)
    assert isinstance(result, prompting.Prompt)
    sections = result.sections
    assert len(sections) == 1
    assert sections[0].title == "Target"
    assert "TestModel Instance" in sections[0].content


def test_utils_prompting_construct_system_prompt_with_context() -> None:
    """Test constructing system prompt with context."""
    target = {"section": "content"}
    context = {"context_key": "context_value"}
    result = prompting.construct_system_prompt(target, context=context)
    assert isinstance(result, prompting.Prompt)
    sections = result.sections
    assert len(sections) == 2
    assert sections[1].title == "Context"
    assert "context_value" in sections[1].content


def test_utils_prompting_construct_system_prompt_with_guardrails() -> None:
    """Test constructing system prompt with guardrails."""
    target = {"section": "content"}
    guardrails = "test guardrails"
    result = prompting.construct_system_prompt(target, guardrails=guardrails)
    assert isinstance(result, prompting.Prompt)
    sections = result.sections
    assert len(sections) == 2
    assert sections[1].title == "Guardrails"
    assert guardrails in sections[1].content


def test_utils_prompting_construct_system_prompt_with_tools() -> None:
    """Test constructing system prompt with tools."""
    target = {"section": "content"}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "A test function",
                "parameters": {
                    "properties": {"param1": {"type": "string", "description": "First parameter"}},
                    "required": ["param1"],
                },
            },
        }
    ]
    result = prompting.construct_system_prompt(target, tools=tools)
    assert isinstance(result, prompting.Prompt)
    sections = result.sections
    assert len(sections) == 2
    assert sections[1].title == "Tools"
    assert "test_func" in sections[1].content
    assert "First parameter" in sections[1].content


if __name__ == "__main__":
    test_utils_prompting_convert_object_to_prompt_context_string()
    test_utils_prompting_convert_object_to_prompt_context_dict()
    test_utils_prompting_convert_object_to_prompt_context_pydantic()
    test_utils_prompting_convert_object_to_prompt_context_markdown()
    test_utils_prompting_convert_object_to_prompt_context_invalid()
    test_utils_prompting_construct_model_prompt_class()
    test_utils_prompting_construct_model_prompt_instance()
    test_utils_prompting_construct_system_prompt_dict()
    test_utils_prompting_construct_system_prompt_model()
    test_utils_prompting_construct_system_prompt_with_context()
    test_utils_prompting_construct_system_prompt_with_guardrails()
    test_utils_prompting_construct_system_prompt_with_tools()
