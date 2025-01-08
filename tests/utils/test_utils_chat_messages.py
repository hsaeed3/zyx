# zyx.utils.chat_messages tests

from typing import Any, List
import pytest
from rich import print
from pydantic import BaseModel
from zyx.types.chat_completions.chat_message import ChatMessage
from zyx.utils import chat_messages as chat_messages_helpers

module_tag = "[bold italic sky_blue3]zyx.utils.chat_messages[/bold italic sky_blue3]"
import logging


logger = logging.getLogger("zyx")


# ===================================================================
# Message Creation
# ===================================================================


def test_utils_chat_messages_create_chat_message() -> None:
    """Test creating chat messages with different inputs."""


    # Test string content
    user_message = chat_messages_helpers.create_chat_message(content="Hello, world!")
    print(f"{module_tag} - [bold green]User Message[/bold green]")
    print(user_message)

    assert "role" in user_message
    assert user_message["role"] == "user"
    assert "content" in user_message
    assert user_message["content"] == "Hello, world!"

    # Test dict content
    dict_message = chat_messages_helpers.create_chat_message({"key": "value"}, role="assistant")
    assert dict_message["role"] == "assistant"
    assert '"key": "value"' in dict_message["content"]

    # Test with tool_call_id
    tool_message = chat_messages_helpers.create_chat_message("Tool response", role="tool", tool_call_id="123")
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "123"

    # Test invalid image role
    with pytest.raises(Exception):
        chat_messages_helpers.create_chat_message("Test", role="assistant", images=["test.jpg"])


def test_utils_chat_messages_validate_chat_message() -> None:
    """Test message validation with different input types."""


    # Test with dict
    valid_dict = {"role": "user", "content": "Hello"}
    validated = chat_messages_helpers.validate_chat_message(valid_dict)
    assert validated["role"] == "user"
    assert validated["content"] == "Hello"

    # Test with Pydantic model
    class TestMessage(BaseModel):
        role: str
        content: str

    model_msg = TestMessage(role="assistant", content="Hi")
    validated = chat_messages_helpers.validate_chat_message(model_msg)
    assert validated["role"] == "assistant"
    assert validated["content"] == "Hi"

    # Test with ChatMessage
    chat_msg = ChatMessage(role="system", content="Be helpful")
    validated = chat_messages_helpers.validate_chat_message(chat_msg)
    assert validated["role"] == "system"

    # Test with invalid dict (missing role)
    invalid_dict = {"content": "Hello"}
    with pytest.raises(Exception):
        chat_messages_helpers.validate_chat_message(invalid_dict)


def test_utils_chat_messages_convert_to_chat_messages() -> None:
    """Test converting different message formats."""


    # Test with string
    messages = chat_messages_helpers.convert_to_chat_messages("Hello")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"

    # Test with single message dict
    single_msg = {"role": "user", "content": "Hi"}
    messages = chat_messages_helpers.convert_to_chat_messages(single_msg)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    # Test with list of messages
    message_list = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    messages = chat_messages_helpers.convert_to_chat_messages(message_list)
    print(f"{module_tag} - [bold green]Messages[/bold green]")
    print(messages)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_utils_chat_messages_format_or_create_system_chat_message() -> None:
    """Test system message formatting and positioning."""


    # Test merging multiple system messages
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Be helpful"},
        {"role": "system", "content": "Be concise"},
    ]
    formatted = chat_messages_helpers.format_or_create_system_chat_message(messages)

    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"] == "Be helpful\nBe concise"
    assert formatted[1]["role"] == "user"

    # Test single system message positioning
    messages = [{"role": "user", "content": "Hi"}, {"role": "system", "content": "Be helpful"}]
    formatted = chat_messages_helpers.format_or_create_system_chat_message(messages)

    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[1]["role"] == "user"

    # Test no system message
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    formatted = chat_messages_helpers.format_or_create_system_chat_message(messages)
    assert len(formatted) == 2
    assert formatted[0]["role"] == "user"
    assert formatted[1]["role"] == "assistant"


def test_utils_chat_messages_add_system_context() -> None:
    """Test adding system context to message threads."""


    # Test adding to thread with no system message
    messages = [{"role": "user", "content": "Hi"}]
    context = "Be helpful"
    formatted = chat_messages_helpers.add_system_context_to_thread(context, messages)
    print(formatted)

    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"] == "Be helpful"
    assert formatted[1]["role"] == "user"

    # Test adding to thread with existing system message
    messages = [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "Hi"}]
    context = "Be helpful"
    formatted = chat_messages_helpers.add_system_context_to_thread(context, messages)
    print(formatted)
    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"] == "Be concise\n\nBe helpful"
    assert formatted[1]["role"] == "user"


def test_utils_chat_messages_add_user_context() -> None:
    """Test adding user context to message threads."""


    # Test adding to thread with no user message
    messages = [{"role": "system", "content": "Be helpful"}]
    context = "Hi there"
    formatted = chat_messages_helpers.add_user_context_to_thread(context, messages)
    assert len(formatted) == 2
    assert formatted[1]["role"] == "user"
    assert formatted[1]["content"] == "Hi there"

    # Test adding to thread with existing user message
    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"}
    ]
    context = "How are you?"
    formatted = chat_messages_helpers.add_user_context_to_thread(context, messages)
    assert len(formatted) == 2
    assert formatted[1]["role"] == "user"
    assert formatted[1]["content"] == "Hello\n\nHow are you?"


if __name__ == "__main__":
    test_utils_chat_messages_create_chat_message()
    test_utils_chat_messages_validate_chat_message()
    test_utils_chat_messages_convert_to_chat_messages()
    test_utils_chat_messages_format_or_create_system_chat_message()
    test_utils_chat_messages_add_system_context()
    test_utils_chat_messages_add_user_context()
