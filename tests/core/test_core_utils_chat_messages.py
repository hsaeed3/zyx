"""
Message helper tests
"""

import pytest
from rich import print
from zyx.core.utils import chat_messages as chat_messages_helpers

module_tag = "[bold italic sky_blue3]zyx.helpers.messages[/bold italic sky_blue3]"


# ===================================================================
# Message Creation
# ===================================================================


def test_core_helpers_chat_messages_create_chat_message():
    # Testing from a string
    user_message = chat_messages_helpers.create_chat_message(content="Hello, world!")
    print(f"{module_tag} - [bold green]User Message[/bold green]")
    print(user_message)

    assert "role" in user_message
    assert user_message["role"] == "user"
    assert "content" in user_message
    assert user_message["content"] == "Hello, world!"


def test_core_helpers_chat_messages_validate_chat_message():
    # Test with dict
    valid_dict = {"role": "user", "content": "Hello"}
    validated = chat_messages_helpers.validate_chat_message(valid_dict)
    assert validated["role"] == "user"
    assert validated["content"] == "Hello"

    # Test with invalid dict (missing role)
    invalid_dict = {"content": "Hello"}
    with pytest.raises(Exception):
        chat_messages_helpers.validate_chat_message(invalid_dict)


def test_core_helpers_chat_messages_convert_to_chat_messages():
    # Test with string
    messages = chat_messages_helpers.convert_to_chat_messages("Hello")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"

    # Test with list of messages
    message_list = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    messages = chat_messages_helpers.convert_to_chat_messages(message_list)
    print(f"{module_tag} - [bold green]Messages[/bold green]")
    print(messages)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_core_helpers_chat_messages_format_or_create_system_chat_message():
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


def test_core_helpers_chat_messages_add_system_context():
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


if __name__ == "__main__":
    test_core_helpers_chat_messages_create_chat_message()
    test_core_helpers_chat_messages_validate_chat_message()
    test_core_helpers_chat_messages_convert_to_chat_messages()
    test_core_helpers_chat_messages_format_or_create_system_chat_message()
    test_core_helpers_chat_messages_add_system_context()
