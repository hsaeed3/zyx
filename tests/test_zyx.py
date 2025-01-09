# full library test runnable
# if you dont want to use pytest, you can run this file directly

import time
from rich import print
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from typing import Callable, List
import logging

from tests import test_logging, test_api_resource
from tests.utils import (
    test_utils_chat_messages,
    test_utils_prompting,
    test_utils_pydantic_models,
    test_utils_tool_calling,
)


logger = logging.getLogger("zyx")
start_time = time.time()


# allow for logging test to clear first
test_logging.test_logging()
logger.setLevel(logging.DEBUG)


# Store all test functions in a list to track count
tests: List[Callable[[], None]] = [
    # ========================================================================
    # [UTILS]
    # ========================================================================
    # chat messages
    test_utils_chat_messages.test_utils_chat_messages_add_system_context,
    test_utils_chat_messages.test_utils_chat_messages_add_user_context,
    test_utils_chat_messages.test_utils_chat_messages_convert_to_chat_messages,
    test_utils_chat_messages.test_utils_chat_messages_create_chat_message,
    test_utils_chat_messages.test_utils_chat_messages_format_or_create_system_chat_message,
    test_utils_chat_messages.test_utils_chat_messages_validate_chat_message,
    # prompting
    test_utils_prompting.test_utils_prompting_convert_object_to_prompt_context_string,
    test_utils_prompting.test_utils_prompting_convert_object_to_prompt_context_dict,
    test_utils_prompting.test_utils_prompting_convert_object_to_prompt_context_pydantic,
    test_utils_prompting.test_utils_prompting_convert_object_to_prompt_context_markdown,
    test_utils_prompting.test_utils_prompting_convert_object_to_prompt_context_invalid,
    test_utils_prompting.test_utils_prompting_construct_model_prompt_class,
    test_utils_prompting.test_utils_prompting_construct_model_prompt_instance,
    test_utils_prompting.test_utils_prompting_construct_system_prompt_dict,
    test_utils_prompting.test_utils_prompting_construct_system_prompt_model,
    test_utils_prompting.test_utils_prompting_construct_system_prompt_with_context,
    test_utils_prompting.test_utils_prompting_construct_system_prompt_with_guardrails,
    test_utils_prompting.test_utils_prompting_construct_system_prompt_with_tools,
    # pydantic models
    test_utils_pydantic_models.test_utils_pydantic_models_parse_string_to_field_mapping,
    test_utils_pydantic_models.test_utils_pydantic_models_parse_type_to_field_mapping,
    test_utils_pydantic_models.test_utils_pydantic_models_convert_to_model_cls,
    test_utils_pydantic_models.test_utils_pydantic_models_convert_function_to_model_cls,
    # tool calling
    test_utils_tool_calling.test_utils_tool_calling_convert_to_openai_tool_from_function,
    test_utils_tool_calling.test_utils_tool_calling_convert_to_openai_tool_from_pydantic,
    test_utils_tool_calling.test_utils_tool_calling_convert_to_openai_tool_from_dict,
    test_utils_tool_calling.test_utils_tool_calling_convert_to_openai_tool_invalid_dict,
]

# Run all tests and track count
num_tests = len(tests)

with Progress(
    BarColumn(),
    TextColumn("[progress.description]{task.description}"),
    TimeElapsedColumn(),
) as progress:
    task = progress.add_task("[cyan]Running tests...", total=num_tests)

    for test in tests:
        test()
        progress.advance(task)

end_time = time.time()
print(f"ZYX TESTS: Total time taken to test {num_tests + 1} modules: {end_time - start_time:.2f} seconds")
