from loguru import logger
from typing import Union
import os


def format_messages(messages: Union[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    """"""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    else:
        return messages


def swap_system_prompt(
    system_prompt: dict = None, messages: Union[str, list[dict[str, str]]] = None
):
    logging_enabled = bool(os.getenv("ASSISTANT_SERVICE_LOGGING"))

    messages = format_messages(messages)

    for message in messages:
        # Check if a system message exists
        if message.get("role") == "system":
            # If a system message exists, swap it with the system_prompt
            message = system_prompt
            # Move the system_prompt to the beginning of the list
            messages.insert(0, message)
            # Remove the system_prompt from its original position
            messages.remove(message)
            break

        else:
            messages.insert(0, system_prompt)
            break

    if logging_enabled:
        logger.info("System prompt swapped.")

    if len([message for message in messages if message.get("role") == "system"]) > 1:
        messages.pop()

    return messages
