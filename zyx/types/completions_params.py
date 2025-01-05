"""
zyx.types.completions_params

This module contains the types for the parameters of the `zyx` completions & chat completions API.
"""

from __future__ import annotations

# [Imports]
from pydantic import BaseModel
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar
from .completions import ChatMessage, ChatModels


# NOTE:
# This typevar follows the `Instructor` pattern to get proper typed
# returns for BaseModel inputs during structured outputs.
T = TypeVar("T", bound=BaseModel)


# ==============================================================
# [Params]
# ==============================================================

CompletionMessagesParam = Union[
    str,
    ChatMessage,
    Dict[str, Any],
    List[Union[ChatMessage, Dict[str, Any]]],
    List[List[Union[ChatMessage, Dict[str, Any]]]],
]
"""
The messages parameter for chat completions.

It can be one of the following types:
- str: A single message as a string.
- ChatMessage: A single message as a ChatMessage object.
- Dict: A single message as a dictionary.
- List[Union[ChatMessage, Dict]]: A list of messages, each as a ChatMessage object or a dictionary.
- List[List[Union[ChatMessage, Dict]]]: A list of lists, where each inner list contains messages as ChatMessage objects or dictionaries.
"""

CompletionModelParam = Union[str, ChatModels, List[Union[str, ChatModels]]]
"""
The model parameter for chat completions.

Provide either a single model name or a list of model names.
The param follows the LiteLLM `model` format, so it uses the:
`provider/model` syntax.
"""

CompletionResponseFormatParam = Union[str, Type, List[Union[str, Type]], Dict[str, Any], T]
"""
The response_format parameter for chat completions.

It can be one of the following types:
- str: A single response format as a string.
- Type: A single response format as a Type object.
- List[Union[str, Type]]: A list of response formats, each as a string or a Type object.
- Dict: A single response format as a dictionary.
- T: A single response format as a TypeVar object.
"""

CompletionToolsParam = Optional[List[Union[Callable, BaseModel, Dict[str, Any]]]]
"""
The tools parameter for chat completions.

It can be one of the following types:
- Optional[List[Union[Callable, BaseModel, Dict[str, Any]]]]: A list of tools, each as a Callable, BaseModel, or a dictionary.
"""
