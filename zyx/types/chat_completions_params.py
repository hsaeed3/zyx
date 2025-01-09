"""
zyx.types.chat_completions_params

Input & output argument types for creating chat completions.
"""

from __future__ import annotations

# [Imports]
from pydantic import BaseModel
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar, Literal

from .chat_completions.chat_message import ChatMessage
from .chat_completions.chat_model import ChatModel


# [TypeVars]
# NOTE:
# This is used to get proper typed returns for BaseModel inputs
# during structured outputs.
T = TypeVar("T", bound=BaseModel)


# ===================================================================
# [Completions & Chat Completions Params]
# ===================================================================

# [Client Provider]
# `client_provider`
ChatCompletionClientProviderParam = Literal["openai", "litellm"]
"""
The base llm client provider used by the `zyx` completions API.

The OpenAI client supports passing in a custom httpx client, so this
is why `zyx` provides dual client support.
"""

# [Completion Messages]
# `messages`
ChatCompletionMessagesParam = Union[
    # Single Message or String Input
    str,
    ChatMessage,
    Dict[str, Any],
    # List of Messages
    List[Union[ChatMessage, Dict[str, Any]]],
    # List of Lists of Messages (Batch)
    List[List[Union[ChatMessage, Dict[str, Any]]]],
]
"""The messages or prompt to use for the completion. Messages
can be provided as any one of:

- str: A single message as a string.
- ChatMessage: A single message as a ChatMessage object.
- Dict[str, Any]: A single message as a dictionary.
- List[Union[ChatMessage, Dict[str, Any]]]: A list of messages.
- List[List[Union[ChatMessage, Dict[str, Any]]]]: A list of lists of messages (batch).
"""

# [Completion Model]
# `model`
ChatCompletionModelParam = Union[
    str,
    ChatModel,
    # List (Batching)
    List[Union[str, ChatModel]],
]
"""The model to use for the completion. Models can be provided as any one of:

- str: A model name as a string.
- List[str]: A list of model names (batch).
"""

# [Completion Response Format]
# `response_format`
ChatCompletionResponseFormatParam = Union[str, Type, List[Union[str, Type]], Dict[str, Any], T]
"""
The response_format parameter for chat completions.

It can be one of the following types:
- str: A single response format as a string.
- Type: A single response format as a Type object.
- List[Union[str, Type]]: A list of response formats, each as a string or a Type object.
- Dict: A single response format as a dictionary.
- T: A single response format as a TypeVar object.
"""

# [Completion Response Format Provider]
# `response_format_provider`
ChatCompletionResponseFormatProviderParam = Literal["instructor", "litellm"]
"""
The provider to use for the response format.

It can be one of the following types:
- "instructor": Use the Instructor provider.
- "litellm": Use the LiteLLM provider.
"""

# [Completion Tools]
# `tools`
ChatCompletionToolsParam = Optional[List[Union[Callable, BaseModel, Dict[str, Any]]]]
"""
The tools parameter for chat completions.

It can be one of the following types:
- Optional[List[Union[Callable, BaseModel, Dict[str, Any]]]]: A list of tools, each as a Callable, BaseModel, or a dictionary.
"""


# ===================================================================
# [Outputs]
# ===================================================================


# [Generic Chat Completion if response_format is provided as a generic type]
TypeCompletion = Union[str, int, float, bool, List[str], List[int], List[float], List[bool]]
"""
A generic type for chat completions that are not zyx specific.
"""
