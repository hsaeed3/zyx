# zyx.messages
# message utilities

from .lib.exception import ZYXException
from .types.completions.message import Message
from typing import List, Union


# formatter
def format_messages(
        messages : Union[
            str,
            Message,
            List[Message],
            List[List[Message]]
        ],
) -> List[Message]:
    
    """
    Formats messages for use in completions.

    Args:
        messages (Union[str, Message, List[Message]]): Messages to format.

    Returns:
        List[Message]: Formatted messages.
    """

    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    
    elif isinstance(messages, Message):
        return [messages]
    
    elif isinstance(messages, list) and isinstance(messages[0], dict):
        return messages
    
    elif isinstance(messages, list) and isinstance(messages[0], list) and isinstance(messages[0][0], dict):
        return messages
    
    else:
        raise ZYXException(f"Invalid message type: {type(messages)}")