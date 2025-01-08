"""
zyx.utils.chat_messages

This module contains helper functions & processors for various tasks such as
message thread formatting, validation, creation, etc.
"""

from __future__ import annotations

# [Imports]
from pathlib import Path
import json
from pydantic import BaseModel
from typing import Any, Optional, Union, Sequence, Dict, List

from ..types.multimodal import Image
from ..types.chat_completions.chat_message import ChatMessage, ChatMessageRole
from zyx import logging


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
            raise logging.ZyxException("Images can only be added to user messages!")

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

        images = processed_images

        logging.logger.debug("processed images", processed_images, f"for use in a {role} chat message")

    # Handle non-string content case
    if not isinstance(content, str):
        content = json.dumps(content)

    # Return message with only present fields
    message_kwargs = {"role": role, "content": content}
    if images:
        message_kwargs["images"] = images
    if tool_call_id:
        message_kwargs["tool_call_id"] = tool_call_id

    message = ChatMessage(**message_kwargs)

    logging.logger.debug("created message", message)

    return message


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
        raise logging.ZyxException(f"Invalid message provided [bold red]no role field found[/bold red], `{message}`.")

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

        return [create_chat_message(content=messages)]

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        message = validate_chat_message(message)

    logging.logger.debug(f"formatted {len(messages)} messages", messages)

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
        logging.logger.debug("no system message found in thread, returning as is")
        return messages

    if len(system_messages) > 1:
        # Create merged system message content
        system_message_content = "\n".join([message.content for message in system_messages])
        system_message = create_chat_message(content=system_message_content, role="system")

        # Remove all previous system messages in thread
        messages = [message for message in messages if message.role != "system"]

        # Add merged system message to the beginning of the thread
        messages.insert(0, system_message)

        logging.logger.debug("merged system messages", system_message, "into one at the beginning of the thread")

    elif len(system_messages) == 1:
        # Validate system message is at the beginning of the thread
        if not messages[0] == system_messages[0]:
            # Remove the existing system message
            messages = [message for message in messages if message.role != "system"]
            # Add the new system message to the beginning of the thread
            messages.insert(0, system_messages[0])

            logging.logger.debug("validated system message position in thread")

    return messages


# ==============================================================
# [Context Injection]
# ==============================================================


def add_system_context_to_thread(
    context: str,
    messages: List[Union[ChatMessage, Dict[str, Any], BaseModel]],
) -> List[ChatMessage]:
    """
    Adds system context to a thread of messages.

    Examples:
        >>> messages = [ChatMessage(role="user", content="Hi")]
        >>> add_system_context_to_thread("Be helpful", messages)
        [ChatMessage(role="system", content="Be helpful"), ChatMessage(role="user", content="Hi")]

    Args:
        context: The system context to add
        messages: The message thread to add context to

    Returns:
        List[ChatMessage]: Messages with system context added
    """
    # Validate all messages are Message objects
    messages = [validate_chat_message(msg) for msg in messages]

    # Determine if system message is present
    if any(message.role == "system" for message in messages):
        # Format thread to validate system message position
        messages = format_or_create_system_chat_message(messages)

        # Build context into system message
        system_content = messages[0]["content"]
        system_content = f"{system_content}\n\n{context}"
        messages[0]["content"] = system_content

    else:
        # Create new system message
        messages.insert(0, create_chat_message(content=context, role="system"))

        logging.logger.debug("added system context to thread as a new system message")

    # Return
    return messages


def add_user_context_to_thread(
    context: str,
    messages: List[Union[ChatMessage, Dict[str, Any], BaseModel]],
) -> List[ChatMessage]:
    """
    Adds user context to a thread of messages.

    Examples:
        >>> messages = [ChatMessage(role="system", content="Be helpful")]
        >>> add_user_context_to_thread("Hi", messages)
        [ChatMessage(role="system", content="Be helpful"), ChatMessage(role="user", content="Hi")]

    Args:
        context: The user context to add
        messages: The message thread to add context to

    Returns:
        List[ChatMessage]: Messages with user context added
    """
    # Validate all messages are Message objects
    messages = [validate_chat_message(msg) for msg in messages]

    # Check if the last message is a user message
    if messages and messages[-1].role == "user":
        # Append context to the last user message
        user_content = messages[-1]["content"]
        user_content = f"{user_content}\n\n{context}"
        messages[-1]["content"] = user_content
    else:
        # Create new user message at the end
        messages.append(create_chat_message(content=context, role="user"))

        logging.logger.debug("added user context to thread as a new user message")

    # Return
    return messages

