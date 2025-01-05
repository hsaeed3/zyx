"""
zyx.core.helpers.chat_messages

This module contains helper functions & processors for various tasks such as
message thread formatting, validation, creation, etc.
"""

from __future__ import annotations

# [Imports]
from pathlib import Path
import json
from pydantic import BaseModel
from typing import Any, Optional, Union, Sequence, Dict, List, Callable

from ....types.completions import ChatMessage, ChatMessageRole, Image
from zyx import utils


# ==============================================================
# [Single Message Creation]
# ==============================================================


def create_chat_message(
    content: Any,
    role: ChatMessageRole = "user",
    images: Optional[Union[Image, Sequence[Image], Path, str]] = None,
    tool_call_id: Optional[str] = None,
) -> ChatMessage:
    """
    Creates a single chat completion message.

    Examples:
        >>> create_chat_message("Hello!", role="user")
        ChatMessage(role="user", content="Hello!")

        >>> create_chat_message("Here's an image", images="path/to/image.jpg")
        ChatMessage(role="user", content="Here's an image", images=[Image(value="path/to/image.jpg")])

    Args:
        content: The message content. Non-string content will be JSON serialized
        role: The role of the message sender (e.g. "user", "assistant", "system")
        images: Optional image(s) to include in the message. Can be:
            - Path to image file
            - URL string
            - Image object
            - List of any of the above

    Returns:
        ChatMessage: A formatted chat completion message

    Raises:
        ZyxException: If images are provided with a non-user role
    """
    # Handle image case
    if images:
        if not role == "user":
            raise utils.ZyxException("Images can only be added to user messages!")

        # Convert to list if single item
        if not isinstance(images, (list, tuple)):
            images = [images]

        # Convert each image to Image object
        processed_images = []
        for img in images:
            if isinstance(img, Image):
                processed_images.append(img)
            else:
                # Path, str (URL or base64), or bytes will be handled by Image model
                processed_images.append(Image(value=img))

        if utils.zyx_debug:
            utils.logger.debug(f"Processed images for {role} message: {processed_images}")

        images = processed_images

    # Handle non-string content case
    if not isinstance(content, str):
        content = json.dumps(content)

    if utils.zyx_debug:
        utils.logger.debug(
            f"Created a role: [plum3]{role}[/plum3] message with content: [italic dim]{content}[/italic dim]"
        )

    # Return message with only present fields
    if images and tool_call_id:
        return ChatMessage(role=role, content=content, images=images, tool_call_id=tool_call_id)
    elif images:
        return ChatMessage(role=role, content=content, images=images)
    elif tool_call_id:
        return ChatMessage(role=role, content=content, tool_call_id=tool_call_id)
    else:
        return ChatMessage(role=role, content=content)


# ==============================================================
# [Single Message Validation]
# ==============================================================


def validate_chat_message(message: Union[ChatMessage, BaseModel, Dict[str, Any]]) -> ChatMessage:
    """
    Validates a single message by ensuring required fields are present and converting to Message type.

    Examples:
        >>> validate_chat_message({"role": "user", "content": "Hello"})
        ChatMessage(role="user", content="Hello")

        >>> validate_chat_message(ChatMessage(role="user", content="Hi"))
        ChatMessage(role="user", content="Hi")

    Args:
        message: The message to validate. Can be a Message object, Pydantic model, or dict

    Returns:
        ChatMessage: A validated ChatMessage object

    Raises:
        ZyxException: If the message is missing the required 'role' field
    """
    if isinstance(message, BaseModel):
        message = message.model_dump()

    if "role" not in message:
        raise utils.ZyxException(f"Invalid message provided [bold red]no role field found[/bold red], `{message}`.")

    if not isinstance(message, ChatMessage):
        return ChatMessage(**message)
    return message


# ==============================================================
# [Thread Conversion & Formatting]
# ==============================================================


def convert_to_chat_messages(
    messages: Union[
        # string messages are converted to one 'role' : 'user' message
        str,
        # standard single message
        Union[ChatMessage, BaseModel, Dict[str, Any]],
        # list of messages
        Sequence[Union[ChatMessage, BaseModel, Dict[str, Any]]],
    ],
) -> List[ChatMessage]:
    """
    Formats input messages into a valid list of OpenAI spec chat completion messages.

    Examples:
        >>> convert_to_chat_messages("Hello")
        [ChatMessage(role="user", content="Hello")]

        >>> convert_to_chat_messages([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])
        [ChatMessage(role="user", content="Hi"), ChatMessage(role="assistant", content="Hello")]

    Args:
        messages: The messages to format. Can be:
            - A single string (converted to user message)
            - A single Message/dict/BaseModel
            - A sequence of Message/dict/BaseModel

    Returns:
        List[ChatMessage]: A list of validated ChatMessage objects
    """
    # clear string case immediately
    if isinstance(messages, str):
        if utils.zyx_debug:
            utils.logger.debug(f"Formatting single string into message thread: [italic dim]{messages}[/italic dim]")

        return [create_chat_message(content=messages)]

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        message = validate_chat_message(message)

    if utils.zyx_debug:
        utils.logger.debug(f"Formatted {len(messages)} messages into message thread.")

    return messages


# ==============================================================
# [System Message Formatting & Creation]
# ==============================================================


def format_or_create_system_chat_message(
    messages: List[Union[ChatMessage, Dict[str, Any], BaseModel]],
) -> List[ChatMessage]:
    """
    Validates and formats system messages within a message thread.

    Examples:
        >>> messages = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="system", content="Be concise")
        ]
        >>> format_or_create_system_chat_message(messages)
        [
            ChatMessage(role="system", content="Be helpful\nBe concise"),
            ChatMessage(role="user", content="Hi")
        ]

    Args:
        messages: List of messages to process. Can be Message objects, dicts, or BaseModels

    Returns:
        List[ChatMessage]: Processed messages with system messages merged and positioned at start
    """
    # Convert all messages to Message objects first
    messages = [validate_chat_message(msg) for msg in messages]

    # [Check for System Message]
    system_messages = [message for message in messages if message.role == "system"]

    if not system_messages:
        if utils.zyx_debug:
            utils.logger.debug("No system message found in thread, returning as is.")

        return messages

    if len(system_messages) > 1:
        # Create merged system message content
        system_message_content = "\n".join([message.content for message in system_messages])
        system_message = create_chat_message(content=system_message_content, role="system")

        # Remove all previous system messages in thread
        messages = [message for message in messages if message.role != "system"]

        # Add merged system message to the beginning of the thread
        messages.insert(0, system_message)

        if utils.zyx_debug:
            utils.logger.debug(
                f"Merged {len(system_messages)} system messages into one at the beginning of the thread."
            )

    elif len(system_messages) == 1:
        # Validate system message is at the beginning of the thread
        if not messages[0] == system_messages[0]:
            # Remove the existing system message
            messages = [message for message in messages if message.role != "system"]
            # Add the new system message to the beginning of the thread
            messages.insert(0, system_messages[0])

            if utils.zyx_debug:
                utils.logger.debug(f"Validated system message position in thread.")

    return messages
