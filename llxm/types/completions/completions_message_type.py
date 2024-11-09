# llxm.types.completions.completions_message_type
# message parameter type

from .completions_message import CompletionsMessage
from typing import Union, List


# base param type
CompletionsMessageType = Union[
    str,
    CompletionsMessage,
    List[CompletionsMessage],
    List[List[CompletionsMessage]]
]