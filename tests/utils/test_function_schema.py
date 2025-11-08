import pytest
from typing import Annotated, Literal, Optional
from dataclasses import dataclass
from pydantic import BaseModel
from zyx.utils.function_schema import get_function_schema, FunctionSchema


# Test basic function schema creation
def test_function_schema_from_simple_function():
    def simple_func(x: int, y: str) -> str:
        """A simple function.

        Args:
            x: An integer parameter
            y: A string parameter
        """
        return f"{x}: {y}"

    schema = get_function_schema(simple_func)
    assert isinstance(schema, FunctionSchema)
    assert schema._function_name == "simple_func"
    assert "x" in schema.schema["properties"]
    assert "y" in schema.schema["properties"]
    assert schema.schema["properties"]["x"]["type"] == "number"
    assert schema.schema["properties"]["y"]["type"] == "string"


def test_function_schema_with_defaults():
    def func_with_defaults(x: int, y: str = "default") -> str:
        """Function with default values."""
        return f"{x}: {y}"

    schema = get_function_schema(func_with_defaults)
    assert "x" in schema.schema["required"]
    assert "y" not in schema.schema["required"]
    assert schema.schema["properties"]["y"]["default"] == "default"


def test_function_schema_with_optional():
    def func_with_optional(x: int, y: Optional[str] = None) -> str:
        """Function with optional parameter."""
        return f"{x}: {y}"

    schema = get_function_schema(func_with_optional)
    assert "x" in schema.schema["required"]


def test_function_schema_with_literal():
    def func_with_literal(x: Literal["a", "b", "c"]) -> str:
        """Function with literal type."""
        return x

    schema = get_function_schema(func_with_literal)
    assert "enum" in schema.schema["properties"]["x"]
    assert set(schema.schema["properties"]["x"]["enum"]) == {"a", "b", "c"}


# Test Pydantic BaseModel schema creation
def test_function_schema_from_basemodel():
    class TestModel(BaseModel):
        """A test model."""

        x: int
        y: str
        z: float = 1.0

    schema = get_function_schema(TestModel)
    assert isinstance(schema, FunctionSchema)
    assert schema._function_name == "TestModel"
    assert "x" in schema.schema["properties"]
    assert "y" in schema.schema["properties"]
    assert "z" in schema.schema["properties"]


def test_function_schema_from_callable_basemodel():
    class CallableModel(BaseModel):
        """A callable model."""

        def __call__(self, x: int, y: str) -> str:
            """Call method.

            Args:
                x: An integer
                y: A string
            """
            return f"{x}: {y}"

    schema = get_function_schema(CallableModel)
    assert isinstance(schema, FunctionSchema)
    assert "x" in schema.schema["properties"]
    assert "y" in schema.schema["properties"]


# Test dataclass schema creation
def test_function_schema_from_dataclass():
    @dataclass
    class TestDataclass:
        """A test dataclass."""

        x: int
        y: str
        z: float = 1.0

    schema = get_function_schema(TestDataclass)
    assert isinstance(schema, FunctionSchema)
    assert schema._function_name == "TestDataclass"
    assert "x" in schema.schema["properties"]
    assert "y" in schema.schema["properties"]
    assert "z" in schema.schema["properties"]


# Test dict schema creation
def test_function_schema_from_dict():
    test_dict = {"x": 1, "y": "test", "z": 1.5}

    schema = get_function_schema(test_dict)
    assert isinstance(schema, FunctionSchema)
    assert "x" in schema.schema["properties"]
    assert "y" in schema.schema["properties"]
    assert "z" in schema.schema["properties"]


# Test takes_context parameter
def test_function_schema_with_takes_context():
    def func_with_context(ctx: dict, x: int, y: str) -> str:
        """Function with context parameter.

        Args:
            ctx: Context dictionary
            x: An integer
            y: A string
        """
        return f"{x}: {y}"

    schema = get_function_schema(func_with_context, takes_context=True)
    assert "ctx" not in schema.schema["properties"]
    assert "x" in schema.schema["properties"]
    assert "y" in schema.schema["properties"]


# Test render_openai_schema
def test_render_openai_schema():
    def test_func(x: int, y: str) -> str:
        """A test function.

        Args:
            x: An integer parameter
            y: A string parameter

        Returns:
            A formatted string
        """
        return f"{x}: {y}"

    schema = get_function_schema(test_func)
    openai_schema = schema.render_openai_schema()

    assert openai_schema["type"] == "function"
    assert "function" in openai_schema
    assert openai_schema["function"]["name"] == "test_func"
    assert "description" in openai_schema["function"]
    assert "parameters" in openai_schema["function"]


def test_render_openai_schema_with_exclude():
    def test_func(x: int, y: str, z: float) -> str:
        """A test function."""
        return f"{x}: {y}: {z}"

    schema = get_function_schema(test_func)
    openai_schema = schema.render_openai_schema(exclude={"z"})

    assert "x" in openai_schema["function"]["parameters"]["properties"]
    assert "y" in openai_schema["function"]["parameters"]["properties"]
    assert "z" not in openai_schema["function"]["parameters"]["properties"]


# Test execute method
def test_execute_function():
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    schema = get_function_schema(add)
    result = schema.execute({"x": 5, "y": 3})
    assert result == 8


def test_execute_with_defaults():
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    schema = get_function_schema(greet)
    result = schema.execute({"name": "World"})
    assert result == "Hello, World!"


def test_execute_missing_required_param():
    def test_func(x: int, y: str) -> str:
        """Test function."""
        return f"{x}: {y}"

    schema = get_function_schema(test_func)
    result = schema.execute({"x": 1}, raise_on_error=False)
    assert "error" in result
    assert "Missing required parameter" in result["error"]


def test_execute_with_error():
    def divide(x: int, y: int) -> float:
        """Divide two numbers."""
        return x / y

    schema = get_function_schema(divide)
    result = schema.execute({"x": 10, "y": 0}, raise_on_error=False)
    assert "error" in result


def test_execute_raises_on_error():
    def divide(x: int, y: int) -> float:
        """Divide two numbers."""
        return x / y

    schema = get_function_schema(divide)
    with pytest.raises(ZeroDivisionError):
        schema.execute({"x": 10, "y": 0}, raise_on_error=True)


# Test model and function accessors
def test_function_accessor():
    def test_func(x: int) -> int:
        """Test function."""
        return x * 2

    schema = get_function_schema(test_func)
    func = schema.function()
    assert callable(func)
    assert func(x=5) == 10


def test_model_accessor():
    def test_func(x: int, y: str) -> str:
        """Test function."""
        return f"{x}: {y}"

    schema = get_function_schema(test_func)
    model = schema.model()
    assert issubclass(model, BaseModel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
