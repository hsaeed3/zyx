"""
zyx.utils.tool_calling

Helper utility for tool calling & conversion using the OpenAI
function calling API specification.
"""

from __future__ import annotations

# [Imports]
import json
from typing import Any, Callable, Dict, Union
from pydantic import BaseModel
from openai import pydantic_function_tool
from ..types.chat_completions.chat_message import ChatMessage
from .pydantic_models import convert_python_function_to_pydantic_model_cls

from zyx import logging


# ==============================================================
# [Converters]
# ==============================================================


def convert_to_openai_tool(tool: Union[Callable, BaseModel, Dict[str, Any]]):
    """
    Converts a Python function or Pydantic model to an OpenAI tool dictionary object.

    Examples:
        >>> convert_to_openai_tool(lambda x: x + 1)
        {"type": "function", "function": {"name": "lambda", "arguments": "x"}}

        >>> convert_to_openai_tool(MyModel(x=1))
        {"type": "function", "function": {"name": "MyModel", "arguments": {"x": 1}}}

    Args:
        tool: The tool to convert. Can be a Python function or Pydantic model.

    Returns:
        Dict[str, Any]: The OpenAI tool dictionary object.
    """
    # Check if tool is already an OpenAI tool dictionary
    if isinstance(tool, Dict):
        if "type" in tool and tool["type"] == "function":
            logging.logger.debug(f"Tool is already an OpenAI tool dictionary: {tool}")
            return tool
        else:
            raise logging.ZyxException(f"Tool is not an OpenAI tool dictionary.")

    # Convert Python function to Pydantic model
    if callable(tool):
        try:
            model = convert_python_function_to_pydantic_model_cls(tool)
            tool = pydantic_function_tool(model)
            logging.logger.debug(f"Converted Python function to OpenAI tool: {tool}")
            return tool
        except Exception as e:
            raise logging.ZyxException(
                f"Failed to convert Python function to OpenAI tool: {e}"
            )

    # Convert Pydantic model to OpenAI tool dictionary
    if isinstance(tool, BaseModel):
        # Ensure the model is a class
        if not isinstance(tool, type):
            tool = tool.__class__
        try:
            tool = pydantic_function_tool(tool)
            logging.logger.debug(f"Converted Pydantic model to OpenAI tool: {tool}")
            return tool
        except Exception as e:
            raise logging.ZyxException(
                f"Failed to convert Pydantic model to OpenAI tool: {e}"
            )


# ==============================================================
# [Tool Output Message]
# ==============================================================


def create_tool_output_message(tool_call: Union[Dict[str, Any], BaseModel], tool_output: Any) -> ChatMessage:
    """
    Creates a tool output message from a tool call and its output.

    Examples:
        >>> tool_call = {"name": "calculator", "arguments": {"x": 1, "y": 2}}
        >>> create_tool_output_message(tool_call, 3)
        Message(role="tool", content="3")

    Args:
        tool_call: The original tool call object
        tool_output: The output returned from executing the tool

    Returns:
        Message: A formatted message containing the tool output
    """

    if isinstance(tool_call, BaseModel):
        tool_call = tool_call.model_dump()

    return ChatMessage(
        role="tool",
        tool_call_id=tool_call["id"],
        content=json.dumps(tool_output),
    )
