# llxm.types.completions.completions_response
# completion response

from pydantic import BaseModel
from openai.types.chat.chat_completion import ChatCompletion
from .completions_response_model_type import CompletionsResponseModelType
from typing import Union, Type


# response
CompletionsResponse = Union[
    # standard completion
    ChatCompletion,
    Type[CompletionsResponseModelType],
    # all structured output formats
    Type[BaseModel],
    str, list[str], int , float, bool, list[int], list[float], list[bool], list
]