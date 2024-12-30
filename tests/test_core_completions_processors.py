"""
completions processors utility tests
"""

# [Imports]
import pytest

from pydantic import BaseModel
from zyx.core.completions.processors import (
    MessagesProcessor,
    PromptingProcessor
)
from zyx.core import utils

# [Set Debug Mode]
utils.set_zyx_debug(True)


# ==============================================================
# [Message Tests]
# ==============================================================

def test_core_completions_processors_messages_processor():
    
    # [Message Creation from String]
    message_from_string = MessagesProcessor.create_message(
        content = "Hello, world!",
        role = "user"
    )
    utils.logger.debug("Message from string:\n" + str(message_from_string) + "\n")
    # [Message Creation w/ Image]
    message_from_image = MessagesProcessor.create_message(
        content = "Hello, world!",
        images = ["https://picsum.photos/200/300"]
    )
    utils.logger.debug("Message from image:\n" + str(message_from_image) + "\n")
    
    # [Message Formatting]
    formatted_message_from_string = MessagesProcessor.format_messages_input(
        messages = "Hello, world!"
    )
    utils.logger.debug("Formatted message from string:\n" + str(formatted_message_from_string) + "\n")
    formatted_message_from_dict = MessagesProcessor.format_messages_input(
        messages = {"role": "user", "content": "Hello, world!"}
    )
    utils.logger.debug("Formatted message from dict:\n" + str(formatted_message_from_dict) + "\n")
    formatted_message_from_list = MessagesProcessor.format_messages_input(
        messages = [
            {"role": "user", "content": "Hello, world!"},
            {"role": "assistant", "content": "Hello, world!"}
        ]
    )
    utils.logger.debug("Formatted message from list:\n" + str(formatted_message_from_list) + "\n")
    
    # [System Message Formatting]
    system_formatted_thread_without_system_prompt = MessagesProcessor.format_system_message_inside_thread(
        messages = [
            {"role": "user", "content": "Hello, world!"},
        ]
    )
    utils.logger.debug("System formatted thread without system prompt:\n" + str(system_formatted_thread_without_system_prompt) + "\n")
    system_formatted_thread_with_system_prompt = MessagesProcessor.format_system_message_inside_thread(
        messages = [
            {"role": "user", "content": "Hello, world!"},
            {"role": "system", "content": "Hello, world!"},
        ]
    )
    utils.logger.debug("System formatted thread with system prompt:\n" + str(system_formatted_thread_with_system_prompt) + "\n")
    
    # [Assertions]
    assert message_from_string.role == "user"
    assert message_from_string.content == "Hello, world!"
    
    assert message_from_image.role == "user"
    assert message_from_image.images is not None
    
    assert isinstance(formatted_message_from_string, list)
    assert formatted_message_from_string[0]["role"] == "user"
    assert formatted_message_from_string[0]["content"] == "Hello, world!"
    
    assert isinstance(formatted_message_from_dict, list)
    assert formatted_message_from_dict[0]["role"] == "user"
    assert formatted_message_from_dict[0]["content"] == "Hello, world!"
    
    assert len(formatted_message_from_list) == 2 
    
    assert "system" not in system_formatted_thread_without_system_prompt[0]["role"]
    
    assert "system" in system_formatted_thread_with_system_prompt[0]["role"]
    assert "system" not in system_formatted_thread_with_system_prompt[1]["role"]
    
    
# [Prompting Tests]
def test_core_completions_processors_prompting_processor():
    
    # [Create Context from BaseModel]
    class ExampleModel(BaseModel):
        name : str
        age : int
        
    example_model = ExampleModel(name = "John", age = 30)
    
    context_from_basemodel_cls = PromptingProcessor.create_string_context_from_object(
        context = ExampleModel
    )
    utils.logger.debug("Context from BaseModel class:\n" + str(context_from_basemodel_cls) + "\n")
    
    # [Create Context from Instantiated Object]
    context_from_basemodel_instance = PromptingProcessor.create_string_context_from_object(
        context = example_model
    )
    utils.logger.debug("Context from BaseModel instance:\n" + str(context_from_basemodel_instance) + "\n")
    
    # [Assertions]
    assert "name" in context_from_basemodel_cls
    assert "age" in context_from_basemodel_cls
    assert "name" in context_from_basemodel_instance
    assert "age" in context_from_basemodel_instance


if __name__ == "__main__":
    test_core_completions_processors_messages_processor()
    test_core_completions_processors_prompting_processor()
    
    