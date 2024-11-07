# zyx.types.completions.completion_response
# completion response

from pydantic import BaseModel
from openai.types.chat.chat_completion import ChatCompletion
from .completion_response_model import CompletionResponseModel
from typing import Union, Type


# response
CompletionResponse = Union[
    # standard completion
    ChatCompletion,
    Type[CompletionResponseModel],
    # all structured output formats
    Type[BaseModel],
    str, list[str], int , float, bool, list[int], list[float], list[bool], list
]
