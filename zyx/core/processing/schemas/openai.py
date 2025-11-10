"""zyx.utils.processing.schemas.openai

OpenAI-compatible function schema generation from ANY Python type.

This module provides a unified interface to convert any Python type
(primitives, classes, functions, Pydantic models, etc.) into OpenAI's
function calling schema format by leveraging pydantic_function_tool.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any

from docstring_parser import parse
from openai.lib._tools import pydantic_function_tool
from pydantic import BaseModel

from .pydantic import to_pydantic_model
from .semantics import to_semantic_description

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_tool_param import (
        ChatCompletionToolParam,
    )

__all__ = ["to_openai_schema"]


def to_openai_schema(
    source: Any,
    name: str | None = None,
    description: str | None = None,
) -> ChatCompletionToolParam:
    """
    Create an OpenAI-compatible function schema from ANY Python type.

    This is the universal entry point for OpenAI schema generation:
    - Functions → schema with parameters from function signature
    - Types (int, str, list[str], etc.) → schema with single parameter of that type
    - Pydantic models → schema from model fields
    - Classes → schema from __init__ parameters
    - Dict objects → schema with fields from dict definition

    The key insight: Everything becomes a Pydantic model first via to_pydantic_model(),
    then we use OpenAI's pydantic_function_tool to generate the schema. This ensures
    consistency and leverages both Pydantic's type handling and OpenAI's schema format.

    Parameters
    ----------
    source : Any
        The source to generate schema for (type, function, model, dict, etc.)
    name : str | None, optional
        Override the function/tool name (defaults to inferred name)
    description : str | None, optional
        Override the description (defaults to docstring or generated description)

    Returns
    -------
    ChatCompletionToolParam
        OpenAI-compatible function schema in ChatCompletionToolParam format:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}

    Examples
    --------
    >>> # From a function
    >>> def search(query: str, limit: int = 10) -> list[str]:
    ...     '''Search for items.'''
    ...     pass
    >>> schema = to_openai_schema(search)
    >>> schema['function']['name']
    'search'

    >>> # From a type
    >>> schema = to_openai_schema(int, name="get_count", description="Get item count")
    >>> schema['function']['parameters']['properties']
    {'value': {'type': 'integer'}}

    >>> # From a Pydantic model
    >>> class User(BaseModel):
    ...     name: str
    ...     age: int
    >>> schema = to_openai_schema(User, name="create_user")
    >>> 'name' in schema['function']['parameters']['properties']
    True

    >>> # From a dict
    >>> schema = to_openai_schema({"name": str, "age": int}, name="create_user")
    >>> 'name' in schema['function']['parameters']['properties']
    True
    """

    # Handle dict sources (not hashable, so can't be cached)
    if isinstance(source, dict):
        # Extract metadata defaults
        _name = name or "Model"
        _description = description

        # Convert source to Pydantic model
        model = to_pydantic_model(
            source, name=_name, description=_description
        )

        # Use OpenAI's pydantic_function_tool to generate the schema
        function_schema = pydantic_function_tool(model)

        # Apply overrides
        if name:
            function_schema["name"] = name
        if description:
            function_schema["description"] = description

        # Generate default description if needed
        if not function_schema.get("description"):
            try:
                function_schema["description"] = to_semantic_description(
                    source
                )
            except Exception:
                function_schema["description"] = f"Schema for {_name}"

        # Wrap in ChatCompletionToolParam format
        return {"type": "function", "function": function_schema}

    # Use cached internal implementation for hashable types
    @functools.lru_cache(maxsize=1000)
    def _cached(src: Any, params: tuple) -> ChatCompletionToolParam:
        """Internal cached implementation. params is (name, description)."""
        _name, _description = params

        # Extract metadata from functions/methods for better defaults
        if inspect.isfunction(src) or inspect.ismethod(src):
            # Use function name and docstring as defaults
            if _name is None:
                _name = src.__name__
            if _description is None:
                docstring = parse(src.__doc__ or "")
                _description = (
                    docstring.short_description
                    or docstring.long_description
                    or ""
                )

        # Convert source to Pydantic model (the universal converter)
        model = to_pydantic_model(
            src, name=_name, description=_description
        )

        # Use OpenAI's pydantic_function_tool to generate the schema
        function_schema = pydantic_function_tool(model)

        # Apply name override if provided
        if _name:
            function_schema["name"] = _name

        # Apply description override if provided
        if _description:
            function_schema["description"] = _description

        # If description is still empty, use semantic description
        if not function_schema.get("description"):
            # Use semantic description for all types
            try:
                function_schema["description"] = to_semantic_description(
                    src
                )
            except Exception:
                # Fallback to simple description
                if isinstance(src, type) and issubclass(src, BaseModel):
                    function_schema["description"] = (
                        f"Schema for {src.__name__}"
                    )
                elif inspect.isfunction(src) or inspect.ismethod(src):
                    function_schema["description"] = f"Call {src.__name__}"
                else:
                    model_name = getattr(model, "__name__", "value")
                    function_schema["description"] = (
                        f"Provide {model_name}"
                    )

        # Enrich with docstring parameter descriptions if source is a function
        if inspect.isfunction(src) or inspect.ismethod(src):
            docstring = parse(src.__doc__ or "")
            param_descriptions = {
                p.arg_name: p.description
                for p in docstring.params
                if p.arg_name and p.description
            }

            # Add parameter descriptions from docstring
            for param_name, desc in param_descriptions.items():
                if (
                    param_name
                    in function_schema["parameters"]["properties"]
                ):
                    if (
                        "description"
                        not in function_schema["parameters"]["properties"][
                            param_name
                        ]
                    ):
                        function_schema["parameters"]["properties"][
                            param_name
                        ]["description"] = desc

        # Wrap in ChatCompletionToolParam format
        return {"type": "function", "function": function_schema}

    return _cached(source, (name, description))
