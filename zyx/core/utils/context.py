"""
zyx.core.logging.context

This module contains helper functions for converting various input object types
to a string context.
"""

from __future__ import annotations

# [Imports]
from typing import Any, Union
from pydantic import BaseModel
import json

from zyx import logging


# ==============================================================
# [Context String Creation]
# ==============================================================


def convert_context_to_string(context: Union[str, BaseModel, Any]) -> str:
    """
    Converts various input object types to a string context.

    Examples:
        >>> convert_context_to_string("Hello")
        "Hello"

        >>> convert_context_to_string({"key": "value"})
        '{"key": "value"}'

    Args:
        context: Object to convert to string context. Can be:
            - String (returned as-is)
            - Pydantic BaseModel class (converted to JSON schema)
            - Pydantic BaseModel instance (converted to JSON)
            - Other objects (converted to JSON string)

    Returns:
        str: The context as a string

    Raises:
        ZyxException: If the object cannot be converted to a string context
    """
    # Return strings as-is
    if isinstance(context, str):
        return context

    # Handle Pydantic model class
    if isinstance(context, type) and issubclass(context, BaseModel):
        try:
            return json.dumps(context.model_json_schema())
        except Exception as e:
            raise logging.ZyxException(f"Failed to get JSON schema from model class {logging.Styles.module(context)}: {e}")

    # Handle Pydantic model instance
    if isinstance(context, BaseModel):
        try:
            return context.model_dump_json()
        except Exception as e:
            raise logging.ZyxException(f"Failed to dump pydantic model {logging.Styles.module(context)} into dict: {e}")

    # Handle other objects by converting to JSON
    try:
        return json.dumps(context)
    except Exception as e:
        raise logging.ZyxException(f"Failed to convert object {logging.Styles.module(context)} to JSON string context: {e}")
