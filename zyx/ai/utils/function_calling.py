"""zyx.ai.utils.function_calling"""

from functools import lru_cache
from inspect import isfunction
from typing import Callable

from pydantic import BaseModel
from openai import pydantic_function_tool
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from .structured_outputs import _convert_function_to_pydantic_basemodel

__all__ = ["openai_function_schema"]


def openai_function_schema(
    function: Callable | BaseModel,
    name: str | None = None,
    description: str | None = None,
) -> ChatCompletionToolParam:
    """Converts a function or a Pydantic model into an appropriate
    OpenAI function schema dictionary representation in the
    `ChatCompletionToolParam` format.

    Args:
        function : The function or Pydantic model to convert.
        name : Optional name for the function.
        description : Optional description for the function.

    Returns:
        A dictionary representing the OpenAI function schema.
    """

    @lru_cache
    def _openai_function_schema(params) -> ChatCompletionToolParam:
        function, name, description = params

        if isfunction(function):
            return pydantic_function_tool(
                _convert_function_to_pydantic_basemodel(function),
                name=name,
                description=description,
            )
        else:
            return pydantic_function_tool(
                function,
                name=name,
                description=description,
            )

    return _openai_function_schema((function, name, description))
