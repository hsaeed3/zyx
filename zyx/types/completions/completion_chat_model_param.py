# completion chat model parameter type

from .completion_chat_model import CompletionChatModel
from typing import Union


# base param type
CompletionChatModelParam = Union[
    str,
    CompletionChatModel
]
