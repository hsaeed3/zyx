"""zyx.processing.toon"""

from __future__ import annotations

import sys
import types
from enum import Enum
from typing import (
    Any,
    Dict,
    Type,
    get_origin,
    get_args,
)

from pydantic import BaseModel
from toon import (
    encode,
    encode_pydantic,
    generate_structure,
    generate_structure_from_pydantic,
)

from ._outputs import is_simple_type

__all__ = ("object_as_toon_text",)


_TYPE_MAPPING: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "float",
    bool: "boolean",
    list: "list",
    tuple: "tuple",
    set: "array",
    frozenset: "array",
}


def _is_type_like(obj: Any) -> bool:
    """
    Check if an object is type-like (includes generic types like Union, Literal, Annotated, etc.).
    """
    if isinstance(obj, type):
        return True
    # Check for Python 3.10+ Union types (int | str)
    if sys.version_info >= (3, 10) and isinstance(obj, types.UnionType):
        return True
    # Check for generic typing constructs (Literal, Annotated, Union, etc.)
    if get_origin(obj) is not None:
        return True
    # If it's considered a simple type, treat it as type-like
    if is_simple_type(obj):
        return True
    return False


def _get_simple_type_toon(obj: type) -> str:
    """
    Get TOON representation for simple types including Unions, Literals,
    Enums, Annotated, and list[Union[...]] patterns.
    """
    from typing import (
        Union as TypingUnion,
        Literal as TypingLiteral,
        Annotated as TypingAnnotated,
    )

    # Direct mapping for basic types
    if obj in _TYPE_MAPPING:
        return _TYPE_MAPPING[obj]

    origin = get_origin(obj)
    args = get_args(obj)

    # Handle Union types at top level (int | str or Union[int, str])
    # Check both typing.Union and types.UnionType (Python 3.10+)
    is_union = False
    if sys.version_info >= (3, 10):
        is_union = (
            isinstance(obj, types.UnionType)
            or origin is type(None)
            or str(origin) == "typing.Union"
        )
    if not is_union and origin is not None:
        is_union = origin is TypingUnion

    if is_union and args:
        # Format union types as "union[type1 | type2 | ...]"
        type_names = []
        for arg in args:
            if arg in _TYPE_MAPPING:
                type_names.append(_TYPE_MAPPING[arg])
            elif arg is type(None):
                type_names.append("null")
            else:
                type_names.append("string")  # fallback for unknown types
        return f"union[{' | '.join(type_names)}]"

    # Handle Literal types: Literal[1, 2, 3] -> literal[1 | 2 | 3]
    if origin is TypingLiteral and args:
        literal_values = [str(arg) for arg in args]
        return f"enum[{' | '.join(literal_values)}]"

    # Handle list with Union types: list[str | int] or list[Union[str, int]]
    if origin in {list, tuple, set}:
        if args:
            inner_arg = args[0]
            inner_origin = get_origin(inner_arg)
            # Check if inner is a Union type (handles both Union and | syntax)
            if inner_origin is not None:
                # Has union or other complex inner type, treat as list
                return _TYPE_MAPPING.get(origin, "list")
        return _TYPE_MAPPING.get(origin, "list")

    # Handle Enum types - format as enum[value1 | value2 | ...]
    if isinstance(obj, type) and issubclass(obj, Enum):
        enum_values = [str(member.value) for member in obj]
        return f"enum[{' | '.join(enum_values)}]"

    # Handle Annotated types - extract the base type
    if origin is TypingAnnotated and args:
        # First arg of Annotated is the actual type
        first_arg = args[0]
        if first_arg in _TYPE_MAPPING:
            return _TYPE_MAPPING[first_arg]
        # Recursively handle complex annotated types
        return _get_simple_type_toon(first_arg)

    # Default fallback for any other generic types
    if origin is not None and args:
        # Try to extract first arg
        first_arg = args[0]
        if first_arg in _TYPE_MAPPING:
            return _TYPE_MAPPING[first_arg]

    # Default fallback
    return "string"


def object_as_toon_text(obj: Any | BaseModel | Type[Any | BaseModel]) -> str:
    """
    Converts a type or value to a TOON string representation, if it
    is supported by the `toonify` library. If it is not supported by `toonify`,
    a custom TOON string representation is generated.

    Args:
        obj : Any | BaseModel | Type[Any | BaseModel]
            The type or value to convert to a TOON string representation.

    Returns:
        str
            The TOON string representation of the type or value.

    Raises:
        ValueError : If the type or value is not supported by the `toonify` library.
    """
    if _is_type_like(obj):
        try:
            if hasattr(obj, "model_fields"):
                return generate_structure_from_pydantic(obj)
            else:
                if is_simple_type(obj):
                    return _get_simple_type_toon(obj)
                else:
                    return generate_structure(obj)  # type: ignore
        except Exception as e:
            raise ValueError(
                f"Failed to generate TOON structure representation for type: {obj}. Error: {e}"
            )

    try:
        if hasattr(obj, "model_fields"):
            return encode_pydantic(obj)
        else:
            return encode(obj)
    except Exception as e:
        raise ValueError(
            f"Failed to generate TOON string representation for value: {obj} of type: {type(obj)}. Error: {e}"
        )
