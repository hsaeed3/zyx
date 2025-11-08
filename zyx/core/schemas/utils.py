"""zyx.core.schemas.utils

'Additional' / helper utilities used in the primary schema
interface and inference method.

This module provides low-level utilities for converting Python types
to JSON schemas, leveraging Pydantic and instructor for the heavy lifting.
"""

from __future__ import annotations

import functools
import inspect
from typing import Annotated, Any, Callable, Dict, List, Type, TypeAlias, Union

import typing_inspect
from docstring_parser import parse
from instructor.dsl.simple_type import is_simple_type

# sorry i know its private
from openai.lib._tools import pydantic_function_tool
from pydantic import BaseModel, TypeAdapter, create_model

__all__ = [
    "guess_json_type",
    "infer_json_schema",
    "create_pydantic_model",
    "openai_function_schema",
    "validate_with_schema",
    "GuessedJsonType",
    "ObjectJsonSchema",
]


GuessedJsonType: TypeAlias = Union[str, List[str], Dict[str, Any], None]
"""Simple alias representation for the return type of `guess_json_type()`."""


ObjectJsonSchema: TypeAlias = Dict[str, Any]
"""Simple alias representation for a JSON schema object."""


@functools.lru_cache(maxsize=1000)
def guess_json_type(T: type) -> GuessedJsonType:
    """Guess the JSON schema type for a given Python type.

    This is heavily inspired by the `guess_json_type` method from
    the `function_schema` library, enhanced with typing_inspect
    for more robust type detection.

    Args:
        T: The Python type to convert.

    Returns:
        A JSON schema type string, list of type strings, dict for Any, or None for NoneType.
    """
    # Special case for Any
    if T is Any:
        return {}

    # Use typing_inspect for robust origin detection
    origin = typing_inspect.get_origin(T)

    # Handle Annotated types
    if origin is Annotated:
        args = typing_inspect.get_args(T)
        if args:
            return guess_json_type(args[0])

    # Handle Union types (including Optional)
    if typing_inspect.is_union_type(T):
        union_args = typing_inspect.get_args(T, evaluate=True)
        # Filter out NoneType
        union_types = [t for t in union_args if t is not type(None)]

        # Collect unique types
        _types = []
        seen = set()
        for union_type in union_types:
            type_val = guess_json_type(union_type)
            # Use string representation for deduplication since dicts aren't hashable
            type_key = str(type_val)
            if type_val is not None and type_key not in seen:
                _types.append(type_val)
                seen.add(type_key)

        if len(_types) == 1:
            return _types[0]
        return _types if _types else None

    # Handle Literal types
    if typing_inspect.is_literal_type(T):
        literal_args = typing_inspect.get_args(T, evaluate=True)
        if literal_args:
            # Get the union of types of all literal values
            type_args = Union[tuple(type(arg) for arg in literal_args)]
            return guess_json_type(type_args)

    # Handle Tuple types (treat as array)
    if typing_inspect.is_tuple_type(T):
        return "array"

    # Handle generic container types
    if typing_inspect.is_generic_type(T):
        # For generic types, check the origin
        if origin is list or origin is tuple:
            return "array"
        elif origin is dict:
            return "object"
        # For other generics, try to infer from origin
        elif origin is not None:
            return guess_json_type(origin)

    # Handle Callable (not directly representable in JSON schema, return None)
    if typing_inspect.is_callable_type(T):
        return None

    # Handle non-type objects
    if not isinstance(T, type):
        return None

    # Handle NoneType explicitly
    if T is type(None):
        return None

    # Check by name for NoneType (for compatibility)
    try:
        if T.__name__ == "NoneType":
            return None
    except AttributeError:
        pass

    # Handle basic types using issubclass (safer with try-except)
    try:
        # Check bool before int (bool is subclass of int)
        if issubclass(T, bool):
            return "boolean"
        if issubclass(T, str):
            return "string"
        if issubclass(T, (float, int)):
            return "number"
        if issubclass(T, list):
            return "array"
        if issubclass(T, dict):
            return "object"
    except TypeError:
        # Not a class, fall through
        pass

    # Handle by name for built-in types (fallback)
    try:
        if T.__name__ == "list":
            return "array"
        if T.__name__ == "dict":
            return "object"
    except AttributeError:
        pass

    # Unknown type
    return None


def infer_json_schema(source: Any) -> ObjectJsonSchema:
    """
    Infer a complete JSON schema from any Python type or value.

    This uses Pydantic's TypeAdapter to generate full JSON schemas
    with proper handling of nested types, constraints, etc.

    Args:
        source: Can be a type, value, Pydantic model, or callable

    Returns:
        A complete JSON schema dict

    Examples:
        >>> infer_json_schema(int)
        {'type': 'integer'}

        >>> infer_json_schema(list[str])
        {'type': 'array', 'items': {'type': 'string'}}

        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> infer_json_schema(User)
        {'type': 'object', 'properties': {...}, ...}
    """
    # If it's already a Pydantic model, use its schema method
    if isinstance(source, type) and issubclass(source, BaseModel):
        return source.model_json_schema()

    # If it's a function, we'll handle this separately (see create_function_schema)
    if inspect.isfunction(source) or inspect.ismethod(source):
        raise ValueError(
            "Use create_function_schema() for functions. "
            "infer_json_schema() is for types and values only."
        )

    # For everything else, use TypeAdapter
    try:
        adapter = TypeAdapter(source)
        return adapter.json_schema()
    except Exception as e:
        # Fallback: try to infer from the type itself
        simple_type = guess_json_type(
            source if isinstance(source, type) else type(source)
        )
        if simple_type is None or simple_type == {}:
            raise ValueError(f"Cannot infer JSON schema from {source}") from e

        if isinstance(simple_type, dict):
            return simple_type
        elif isinstance(simple_type, list):
            return {"anyOf": [{"type": t} for t in simple_type]}
        else:
            return {"type": simple_type}


def create_pydantic_model(
    source: Any, name: str | None = None, key: str | None = None
) -> Type[BaseModel]:
    """
    Create a Pydantic model from any Python type, value, or function.

    This is the UNIVERSAL entry point - everything becomes a Pydantic model:
    - Types → model with that type
    - Dicts → model with those fields
    - Functions → model of function parameters
    - Pydantic models → return as-is

    Args:
        source: Can be a type, value, dict, function, etc.
        name: Optional name for the generated model

    Returns:
        A Pydantic BaseModel class

    Examples:
        >>> # Type
        >>> Model = create_pydantic_model(int)
        >>> Model(value=42)

        >>> # Dict
        >>> Model = create_pydantic_model({"name": str, "age": int})
        >>> Model(name="John", age=30)

        >>> # Function
        >>> def search(query: str) -> list[str]: pass
        >>> Model = create_pydantic_model(search)
        >>> Model(query="test")
    """

    # If already a Pydantic model, return as-is
    if isinstance(source, type) and issubclass(source, BaseModel):
        return source

    # If it's a function, convert it to a parameter model
    if inspect.isfunction(source) or inspect.ismethod(source):
        return _function_to_pydantic_model(source, name=name)

    # Use instructor's ModelAdapter/prepare_response_model
    try:
        from instructor.dsl.simple_type import ModelAdapter, is_simple_type

        # Check for simple types first (includes Literal, primitives, etc.)
        if is_simple_type(source):
            model = ModelAdapter[source]

            # sadly we have to do some attr stuff
            if key:
                setattr(model, key, model.content)
                delattr(model, "content")
        else:
            # For complex types, use prepare_response_model
            from instructor.processing.response import prepare_response_model

            model = prepare_response_model(source)

        if name and hasattr(model, "__name__"):
            model.__name__ = name
        return model
    except Exception as e:
        raise ValueError(f"Cannot create Pydantic model from {source}") from e


def _snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _function_to_pydantic_model(
    func: Callable[..., Any],
    name: str | None = None,
    skip_first_param: bool = False,
) -> Type[BaseModel]:
    """
    Convert a function to a Pydantic model of its parameters.

    This is the key: functions become models so they can flow through
    the same pipeline as everything else.

    Args:
        func: The function to convert
        name: Optional name for the model (defaults to PascalCase of func.__name__)
        skip_first_param: Skip first param (for methods with self/cls)

    Returns:
        A Pydantic model with fields matching the function parameters
    """
    sig = inspect.signature(func)
    docstring = parse(func.__doc__ or "")

    # Build field definitions
    fields: dict[str, Any] = {}
    params = list(sig.parameters.items())

    if skip_first_param and params:
        params = params[1:]

    for param_name, param in params:
        # Get type annotation
        annotation = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any
        )

        # Get description from docstring
        param_doc = next(
            (p.description for p in docstring.params if p.arg_name == param_name), None
        )

        # Build field with description
        if param.default != inspect.Parameter.empty:
            # Has default value
            from pydantic import Field

            fields[param_name] = (
                annotation,
                Field(default=param.default, description=param_doc),
            )
        else:
            # Required field
            from pydantic import Field

            fields[param_name] = (
                annotation,
                Field(description=param_doc) if param_doc else ...,
            )

    # Create the model with PascalCase name for the Pydantic model
    model_name = name or _snake_to_pascal(func.__name__)
    return create_model(model_name, **fields)  # type: ignore


@functools.lru_cache(maxsize=1000)
def openai_function_schema(
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    skip_first_param: bool = False,
) -> dict[str, Any]:
    """
    Create an OpenAI-compatible function schema from a Python function.

    This extracts parameter types and docstrings to generate a complete
    function schema that can be passed to LLMs as a tool.

    Args:
        func: The function to generate schema for
        name: Override the function name (defaults to func.__name__)
        description: Override the description (defaults to docstring)
        skip_first_param: If True, skip the first parameter (e.g., for methods with 'self')

    Returns:
        OpenAI-compatible function schema dict with name, description, and parameters

    Example:
        >>> def search(query: str, limit: int = 10) -> list[str]:
        ...     '''Search for items.
        ...
        ...     Args:
        ...         query: The search query
        ...         limit: Maximum number of results
        ...     '''
        ...     pass
        >>> schema = create_function_schema(search)
        >>> schema['name']
        'search'
        >>> schema['parameters']['properties']['query']
        {'type': 'string', 'description': 'The search query'}
    """
    # Get function signature
    sig = inspect.signature(func)
    params = list(sig.parameters.items())

    # Skip first parameter if requested (for methods)
    if skip_first_param and params:
        params = params[1:]

    # Parse docstring for descriptions
    docstring = parse(func.__doc__ or "")
    param_descriptions = {
        p.arg_name: p.description
        for p in docstring.params
        if p.arg_name and p.description
    }

    # Build field definitions for Pydantic model
    field_definitions: dict[str, Any] = {}
    for param_name, param in params:
        # Get type annotation
        if param.annotation is inspect.Parameter.empty:
            param_type = Any
        else:
            param_type = param.annotation

        # Get default value
        if param.default is inspect.Parameter.empty:
            field_default = ...  # Required field
        else:
            field_default = param.default

        field_definitions[param_name] = (param_type, field_default)

    # Create a temporary Pydantic model
    model_name = name or func.__name__
    temp_model = create_model(
        model_name, __doc__=description or func.__doc__ or "", **field_definitions
    )

    function_schema = pydantic_function_tool(temp_model)

    # Enrich with docstring parameter descriptions
    for param_name, desc in param_descriptions.items():
        if param_name in function_schema["parameters"]["properties"]:
            if (
                "description"
                not in function_schema["parameters"]["properties"][param_name]
            ):
                function_schema["parameters"]["properties"][param_name][
                    "description"
                ] = desc

    # Override name if provided
    if name:
        function_schema["name"] = name

    # Override description if provided
    if description:
        function_schema["description"] = description

    # Wrap in ChatCompletionToolParam format
    return {"type": "function", "function": function_schema}


def validate_with_schema(data: Any, source: Any) -> Any:
    """
    Validate data against a schema defined by a source type.

    This creates a Pydantic model from the source and validates
    the data against it, returning the validated/parsed result.

    Args:
        data: The data to validate (can be dict, primitive, etc.)
        source: The type/schema to validate against (type, BaseModel, function, etc.)

    Returns:
        The validated data, typed according to the source

    Raises:
        ValidationError: If the data doesn't match the schema

    Examples:
        >>> validate_with_schema({"name": "John", "age": 30}, User)
        User(name='John', age=30)

        >>> validate_with_schema(42, int)
        42

        >>> validate_with_schema("invalid", int)
        ValidationError: ...
    """
    # If source is a function, we can't validate against it directly
    if inspect.isfunction(source) or inspect.ismethod(source):
        raise ValueError(
            "Cannot validate data against a function. "
            "Validate against the function's parameter types instead."
        )

    # Special case: if source is a simple Python type, use TypeAdapter directly
    if is_simple_type(source):
        adapter = TypeAdapter(source)
        return adapter.validate_python(data)

    # If source is already a Pydantic model, use it directly
    if isinstance(source, type) and issubclass(source, BaseModel):
        model = source
    else:
        # Create a Pydantic model from the source
        model = create_pydantic_model(source)

    # Validate the data
    if isinstance(data, dict):
        # Data is a dict, validate as keyword arguments
        validated = model(**data)
    else:
        # Data is a single value
        # For simple types wrapped by instructor, try to find the wrapper field
        fields = list(model.model_fields.keys())
        if len(fields) == 1:
            # Single field model (likely a wrapped primitive)
            field_name = fields[0]
            validated = model(**{field_name: data})
            # Extract the wrapped value
            return getattr(validated, field_name)
        else:
            # Multi-field model, data should be a dict
            raise TypeError(
                f"Expected dict for model {model.__name__} with fields {fields}, "
                f"got {type(data).__name__}"
            )

    # Return the validated result (for dict inputs)
    return validated
