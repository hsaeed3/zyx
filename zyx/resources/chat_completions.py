"""
### zyx.resources.chat_completions

Primary resource for the `ChatCompletions` API and `chat_completion` method. This method is the bottom level
resource for all other chat completion based methods in the `zyx` package.
"""

from __future__ import annotations

# [Imports]
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, TypeVar, Type
from ..types.chat_completions.chat_completion import ChatCompletion
from ..types.chat_completions.chat_message import ChatMessage
from ..types.chat_completions_params import (
    ChatCompletionClientProviderParam,
    ChatCompletionMessagesParam,
    ChatCompletionModelParam,
    ChatCompletionResponseFormatParam,
    ChatCompletionResponseFormatProviderParam,
    ChatCompletionToolsParam,
    # Outputs
    TypeCompletion,
)

from ..utils.chat_messages import convert_to_chat_messages
from ..utils.pydantic_models import convert_to_pydantic_model_cls
from ..utils.tool_calling import convert_to_openai_tool
from .._api_resource import APIResource, LiteLLM
from openai import OpenAI
from zyx import logging


# [Types]
# Used for specific typing of the `response_format` parameter if passed as a basemodel
T = TypeVar("T", bound=BaseModel)


# [Exception]
ZyxChatCompletionsException = logging.ZyxException


# ===================================================================
# [Config]
# ===================================================================


class ChatCompletionConfig(BaseModel):
    """
    Extended configuration for a single chat completion.
    """

    # Batching
    batch_models: bool = False
    """If multiple models are provided"""
    batch_messages: bool = False
    """If multiple threads of messages are provided"""
    # Output Format Helper Flag
    has_response_format_as_type: Optional[Type] = None
    """If the response format was provided as a generic type (str, int) (Not a pydantic model class)"""
    response_format_provider: str = None
    """If the user has specifically selected either response_format_provider : 'litellm' or 'instructor'"""


# ===================================================================
# [Handler]
# ===================================================================


class ChatCompletionResource(APIResource):
    """
    The handler for creating single chat completions through the extended
    `zyx` completions API.
    """

    @staticmethod
    def _create_chat_completion_config(
        messages: ChatCompletionMessagesParam,
        model: ChatCompletionModelParam,
        response_format: Optional[ChatCompletionResponseFormatParam] = None,
        response_format_provider: Optional[ChatCompletionResponseFormatProviderParam] = None,
    ) -> ChatCompletionConfig:
        config = ChatCompletionConfig()

        # Batching
        if isinstance(messages, list) and isinstance(messages[0], list):
            config.batch_messages = True
        elif isinstance(messages, list) and isinstance(messages[0], str):
            config.batch_messages = False
        else:
            raise ZyxChatCompletionsException(f"Invalid messages parameter provided, {messages}")

        # Model Batching
        if isinstance(model, list) and isinstance(model[0], str):
            config.batch_models = True
        elif isinstance(model, str):
            config.batch_models = False
        else:
            raise ZyxChatCompletionsException(f"Invalid model parameter provided, {model}")

        # Response Format
        if response_format is not None:
            if isinstance(response_format, type) and not issubclass(response_format, BaseModel):
                config.has_response_format_as_type = True
            else:
                config.has_response_format_as_type = False

            if response_format_provider is None:
                config.response_format_provider = "instructor"
            else:
                config.response_format_provider = response_format_provider

        return config

    @staticmethod
    def _preprocess_messages_param(
        config: ChatCompletionConfig,
        messages: ChatCompletionMessagesParam,
    ) -> Union[List[ChatMessage], List[List[ChatMessage]]]:
        """
        Preprocesses messages of different types into an OpenAI compatible specification.

        Args:
            config (ChatCompletionsConfig): The configuration for the chat completions.
            messages (ChatCompletionMessagesParam): The messages to preprocess.

        Returns:
            Union[List[ChatMessage], List[List[ChatMessage]]]: The preprocessed messages.
        """
        try:
            if config.batch_messages:
                return [convert_to_chat_messages(message) for message in messages]
            else:
                return convert_to_chat_messages(messages)
        except Exception as e:
            raise ZyxChatCompletionsException(
                f"Failed to preprocess messages from provided parameter, {messages} \n: {e}"
            )

    @staticmethod
    def _preprocess_prompting_params(
        config: ChatCompletionConfig,
        messages: Union[List[ChatMessage], List[List[ChatMessage]]],
        system: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> Union[List[ChatMessage], List[List[ChatMessage]]]:
        """
        Preprocessed any applicable prompting parameters into a proper
        system prompt.
        """

    @staticmethod
    def _preprocess_response_format_param(
        config: ChatCompletionConfig,
        response_format: ChatCompletionResponseFormatParam,
    ) -> Type[BaseModel]:
        """
        Processes the response_format parameter properly into a pydantic model.
        """
        try:
            return convert_to_pydantic_model_cls(response_format)
        except Exception as e:
            raise ZyxChatCompletionsException(
                f"Failed to preprocess response_format into a pydantic model class from provided parameter, {response_format} \n: {e}"
            )

    @staticmethod
    def _preprocess_tools_param(tools: Optional[ChatCompletionToolsParam] = None) -> List[Dict[str, Any]]:
        """
        Preprocesses the tools parameter into a list of tool dictionaries.
        """
        if tools is None:
            return None
        for tool in tools:
            if (
                isinstance(tool, dict)
                or isinstance(tool, BaseModel)
                and hasattr(tool, "type")
                and tool["type"] == "function"
            ):
                pass
            else:
                try:
                    tool = convert_to_openai_tool(tool)
                except Exception as e:
                    raise ZyxChatCompletionsException(
                        f"Failed to preprocess tool from provided parameter, {tool} \n: {e}"
                    )
        return tools
