"""
zyx.core.utils.pydantic_models

This module contains various functions & processors used for interacting with
Pydantic models throughout the `zyx` package.
"""

from __future__ import annotations

# [Imports]
from typing import Any, Dict, Optional, Sequence, Union, Type, Callable, get_type_hints
from pydantic import BaseModel, Field, create_model
import docstring_parser
from inspect import signature

from zyx import logging


# ==============================================================
# [Field Mapping Creation]
# ==============================================================


def parse_string_to_pydantic_field_mapping(model_string: str) -> Dict[str, Any]:
    """
    Creates a field mapping from a string specification.

    Parses field definitions in the format:
    ```field : type```

    Supports basic types (str, int, float, bool) and nested types (list, dict).
    If no type is provided, defaults to string.

    Args:
        model_string: String specification of field and type

    Returns:
        Dict mapping field names to (type, Field) tuples
    """
    try:
        field_mapping = {}

        # Handle case with no type specified
        if ":" not in model_string:
            field = model_string.strip()
            # If field starts with capital, assume it's a model name
            # Create a default "value" field
            if field[0].isupper():
                field_mapping["value"] = (str, Field())
            else:
                field_mapping[field] = (str, Field())
            return field_mapping

        # Split into field and type
        field, type_str = model_string.split(":", 1)
        field = field.strip()
        type_str = type_str.strip().lower()

        # Map type string to Python type
        type_mapping = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": float,
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }

        field_type = type_mapping.get(type_str, str)
        field_mapping[field] = (field_type, Field())

    except Exception as e:
        raise logging.ZyxException(f"Error creating field mapping from string: {e}")

    return field_mapping


def parse_type_to_pydantic_field_mapping(type_hint: Type, index: Optional[int] = None) -> Dict[str, Any]:
    """
    Creates a field mapping from a type hint.

    Maps Python types to OpenAI-compatible field definitions.
    Uses indexed field names to avoid collisions.

    Args:
        type_hint: Python type to convert
        index: Optional index to append to field name

    Returns:
        Dict with uniquely named field mapped to type
    """
    name_mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
        tuple: "array",
        set: "array",
        Any: "string",
    }

    # Create unique field name based on type and index
    base_name = name_mapping.get(type_hint, "value").lower()
    field_name = f"{base_name}_{index}" if index is not None else base_name

    return {field_name: (type_hint, Field(...))}


# ==============================================================
# [Model Creation // Conversion]
# ==============================================================


def convert_python_function_to_pydantic_model_cls(func: Callable, model_name: Optional[str] = None) -> Type[BaseModel]:
    """
    Converts a Python function to a Pydantic BaseModel class.

    Extracts type hints, default values, and docstring metadata to create a
    properly annotated Pydantic model representing the function's parameters.

    Args:
        func: The function to convert to a BaseModel
        model_name: Optional name for the generated model class. Defaults to function name.

    Returns:
        Type[BaseModel]: A new Pydantic BaseModel class representing the function parameters

    Raises:
        ValueError: If the function has invalid type hints or docstring
    """
    # Use function name if no model name provided
    if not model_name:
        model_name = func.__name__

    # Get type hints and signature
    hints = get_type_hints(func)
    sig = signature(func)

    # Parse docstring
    docstring = docstring_parser.parse(func.__doc__ or "")

    # Build field definitions
    fields = {}
    for name, param in sig.parameters.items():
        # Get type annotation
        field_type = hints.get(name, Any)

        # Get default value
        default = ... if param.default is param.empty else param.default

        # Find parameter description from docstring
        description = ""
        for doc_param in docstring.params:
            if doc_param.arg_name == name:
                description = doc_param.description
                break

        # Create field with type, default and description
        fields[name] = (field_type, Field(default=default, description=description))

    # Create model class
    model = create_model(model_name, __doc__=docstring.short_description, **fields)

    logging.logger.debug(f"Created Pydantic model class from function {func.__name__}: {model.model_json_schema()}")

    return model


def convert_to_pydantic_model_cls(
    target: Union[str, Type, Sequence[Union[str, Type]], Dict[str, Any], BaseModel],
) -> Type[BaseModel]:
    """
    Creates a Pydantic model class from various input formats.

    Handles:
    - Single field strings ("name: str")
    - Multiple field strings (["name: str", "age: int"])
    - Python types (str, int, etc)
    - Mixed type sequences ([int, "name: str", str])
    - Existing Pydantic models
    - Dict specifications

    Args:
        target: Format specification

    Returns:
        Pydantic model for response validation
    """
    # Handle single string
    if isinstance(target, str):
        field_mapping = parse_string_to_pydantic_field_mapping(target)
        # Use capitalized string as model name if provided
        model_name = target.strip() if target.strip()[0].isupper() else "Response"
        model = create_model(model_name, **field_mapping)


        logging.logger.debug(f"Converted string {target} to Pydantic model class: {model.model_json_schema()}")

        return model

    # Handle type hint
    if isinstance(target, type) and issubclass(target, BaseModel):
        # Return the model class directly instead of creating a new one

        logging.logger.debug(f"Returning class type for initialized Pydantic model: {target.model_json_schema()}")

        return target

    elif isinstance(target, type):
        field_mapping = parse_type_to_pydantic_field_mapping(target)

        model = create_model("Response", **field_mapping)

        logging.logger.debug(f"Converted generic type {target} to Pydantic model class: {model.model_json_schema()}")

        return model

    # Handle list/sequence of fields
    elif isinstance(target, (list, tuple)):
        field_mapping = {}
        type_hint_count = 0  # Counter for type hint fields

        # Check if first item is capitalized string for model name
        model_name = "Response"
        if isinstance(target[0], str) and target[0].strip()[0].isupper():
            model_name = target[0].strip()

        for field_spec in target:
            if isinstance(field_spec, str):
                field_mapping.update(parse_string_to_pydantic_field_mapping(field_spec))
            elif isinstance(field_spec, type):
                field_mapping.update(parse_type_to_pydantic_field_mapping(field_spec, index=type_hint_count))
                type_hint_count += 1

        model = create_model(model_name, **field_mapping)

        logging.logger.debug(f"Converted {target} to Pydantic model class: {model.model_json_schema()}")

        return model

    elif isinstance(target, BaseModel) and not isinstance(target, type):
        logging.logger.debug(f"Returning class type for initialized Pydantic model: {target.model_json_schema()}")

        return target.__class__

    # Handle dict specification
    elif isinstance(target, dict):
        model = create_model("Response", **target)

        logging.logger.debug(f"Converted dictionary {target} to Pydantic model class: {model.model_json_schema()}")

        return model

    else:
        raise logging.ZyxException(
            f"Invalid response format type: {type(target)}. " "Must be string, type, sequence, dict or Pydantic model."
        )
