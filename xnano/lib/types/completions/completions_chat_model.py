# completion chat model parameter type

from .completions_chat_models import CompletionsChatModels
from typing import Union


# base param type
CompletionsChatModel = Union[
    str,
    CompletionsChatModels
]
