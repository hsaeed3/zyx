"""zyx.processing.outputs"""

from __future__ import annotations

import dataclasses
from enum import Enum
from inspect import isclass
from functools import lru_cache
import typing

from pydantic import BaseModel, Field, create_model

__all__ = (
    "is_simple_type",
    "normalize_output_type",
)


def is_simple_type(
    obj: type[BaseModel] | str | int | float | bool | typing.Any,
) -> bool:
    """
    Helper method to determine if a given object is a simple type.

    NOTE: This code is taken directly from `instructor.dsl.simple_type`, you
    can view the original code here:
    [code](https://github.com/567-labs/instructor/blob/main/instructor/dsl/simple_type.py)
    """
    # Get the origin of the response model
    origin = typing.get_origin(obj)

    # Handle special case for list[int | str], list[Union[int, str]] or similar type patterns
    # Identify a list type by checking for various origins it might have
    if origin in {typing.Iterable, list}:
        # For list types, check the contents before deciding
        if origin is list:
            # Extract the inner types from the list
            args = typing.get_args(obj)
            if args and len(args) == 1:
                inner_arg = args[0]
                # Special handling for Union types
                inner_origin = typing.get_origin(inner_arg)

                # Explicit check for Union types - try different patterns across Python versions
                if (
                    inner_origin is typing.Union
                    or inner_origin == typing.Union
                    or str(inner_origin) == "typing.Union"
                    or str(type(inner_arg)) == "<class 'typing._UnionGenericAlias'>"
                ):
                    return True

                # Check for Python 3.10+ pipe syntax
                if hasattr(inner_arg, "__or__"):
                    return True

                # For simple list with basic types, also return True
                if inner_arg in {str, int, float, bool}:
                    return True

                # Check if inner type is a BaseModel - if so, not a simple type
                try:
                    if isclass(inner_arg) and issubclass(inner_arg, BaseModel):
                        return False
                except TypeError:
                    pass

            # If no args or unknown pattern, treat as simple list
            return len(args) == 0

        # Extract the inner types from the list for other iterable types
        args = typing.get_args(obj)
        if args and len(args) == 1:
            inner_arg = args[0]
            # Special handling for Union types
            inner_origin = typing.get_origin(inner_arg)

            # Explicit check for Union types - try different patterns across Python versions
            if (
                inner_origin is typing.Union
                or inner_origin == typing.Union
                or str(inner_origin) == "typing.Union"
                or str(type(inner_arg)) == "<class 'typing._UnionGenericAlias'>"
            ):
                return True

            # Check for Python 3.10+ pipe syntax
            if hasattr(inner_arg, "__or__"):
                return True

            # For simple list with basic types, also return True
            if inner_arg in {str, int, float, bool}:
                return True

        # For other iterable patterns, return False (e.g., streaming types)
        return False

    if obj in {
        str,
        int,
        float,
        bool,
    }:
        return True

    # If the obj is a simple type like annotated
    if origin in {
        typing.Annotated,
        typing.Literal,
        typing.Union,
        list,  # origin of List[T] is list
    }:
        return True

    if isclass(obj) and issubclass(obj, Enum):
        return True

    return False


@lru_cache(maxsize=128)
def _normalize_hashable_output_type(
    obj: typing.Any | type[typing.Any],
) -> type[typing.Any]:
    if is_simple_type(obj):
        return obj

    if isinstance(obj, type) and issubclass(obj, BaseModel):
        return obj

    if dataclasses.is_dataclass(obj) and isinstance(obj, type):
        field_defs: dict[str, tuple[typing.Any, typing.Any]] = {}
        for dc_field in dataclasses.fields(obj):
            annotation = dc_field.type if dc_field.type is not None else typing.Any
            if (
                dc_field.default is dataclasses.MISSING
                and dc_field.default_factory is dataclasses.MISSING
            ):
                default: typing.Any = ...
            elif dc_field.default_factory is not dataclasses.MISSING:
                default = Field(default_factory=dc_field.default_factory)
            else:
                default = Field(default=dc_field.default)
            field_defs[dc_field.name] = (annotation, default)
        model = create_model(obj.__name__ or "Response", **field_defs)  # type: ignore[call-overload]
        if obj.__doc__:
            model.__doc__ = obj.__doc__
        return model

    if isinstance(obj, type):
        return obj

    return type(obj)


def normalize_output_type(
    obj: typing.Any | type[typing.Any],
) -> type[typing.Any]:
    try:
        return _normalize_hashable_output_type(obj)
    except TypeError:
        if isinstance(obj, dict):
            field_defs: dict[str, tuple[typing.Any, typing.Any]] = {}
            for field_name, value in obj.items():
                if isinstance(value, type) or is_simple_type(value):
                    field_defs[field_name] = (value, ...)
                else:
                    field_defs[field_name] = (type(value), Field(default=value))
            return create_model("Response", **field_defs)  # type: ignore[call-overload]

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return _normalize_hashable_output_type(type(obj))

        return type(obj)
