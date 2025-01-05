"""
zyx.core.helpers.prompting

This module is an `extension` of the core `chat_messages` helper, providing
functions specifically geared towards prompting (adding context, instructions)
as well as a few general purpose functions directly for system messages.
"""

from __future__ import annotations

# [Imports]
from typing import Any, Union
from pydantic import BaseModel
import json

from zyx import utils


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
    context_string = None

    # Create context string from pydantic models
    if isinstance(context, type) and issubclass(context, BaseModel):
        try:
            context_string = context.model_json_schema()
        except Exception as e:
            raise utils.ZyxException(f"Failed to get JSON schema from model class {utils.Styles.module(context)}: {e}")
    elif isinstance(context, BaseModel):
        try:
            context_string = context.model_dump_json()
        except Exception as e:
            raise utils.ZyxException(f"Failed to dump pydantic model {utils.Styles.module(context)} into dict: {e}")

    # Convert to JSON string if not already
    if not isinstance(context, str):
        try:
            context_string = json.dumps(context)
        except Exception as e:
            raise utils.ZyxException(
                f"Failed to convert object {utils.Styles.module(context)} to JSON string context: {e}"
            )

    if utils.zyx_debug:
        utils.logger.debug(f"Converted object of type: [italic]{type(context)} {utils.Styles.module(context_string)}.")

    return context_string
