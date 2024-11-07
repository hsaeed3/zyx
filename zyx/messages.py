# zyx.messages
# message utilities

from .lib.exception import ZYXException
from .types.completions.completion_message import CompletionMessage
from typing import List, Union


# formatter
def format_messages(
        messages : Union[
            str,
            CompletionMessage,
            List[CompletionMessage],
            List[List[CompletionMessage]]
        ],
) -> List[CompletionMessage]:
    
    """
    Formats messages for use in completions.

    Args:
        messages (Union[str, CompletionMessage, List[CompletionMessage]]): Messages to format.

    Returns:
        List[CompletionMessage]: Formatted messages.
    """

    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    
    elif isinstance(messages, CompletionMessage):
        return [messages]
    
    elif isinstance(messages, list) and isinstance(messages[0], dict):
        return messages
    
    elif isinstance(messages, list) and isinstance(messages[0], list) and isinstance(messages[0][0], dict):
        return messages
    
    else:
        raise ZYXException(f"Invalid message type: {type(messages)}")