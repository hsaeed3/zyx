# zyx.types.completions.completion_message_param
# message parameter type

from .completion_message import CompletionMessage
from typing import Union, List


# base param type
CompletionMessageParam = Union[
    str,
    CompletionMessage,
    List[CompletionMessage],
    List[List[CompletionMessage]]
]
