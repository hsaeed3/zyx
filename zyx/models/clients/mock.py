"""zyx.models.clients.mock"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from datetime import datetime
from enum import Enum
from typing import Any, List, Type, TypeVar, get_args, get_origin

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import (
    CreateEmbeddingResponse as EmbeddingModelResponse,
)
from openai.types.create_embedding_response import Embedding
from pydantic import BaseModel

from . import ModelClient

__all__ = ["MockModelClient", "MockAsyncOpenAI"]


T = TypeVar("T")


def _generate_mock_value(field_type: Type[Any]) -> Any:
    """Generate a mock value for a given type."""
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Optional types
    if origin is type(None) or field_type is type(None):
        return None

    # Handle Union types (including Optional which is Union[X, None])
    if origin is type(lambda: None) or str(origin) == "typing.Union":
        # Get first non-None type
        for arg in args:
            if arg is not type(None):
                return _generate_mock_value(arg)
        return None

    # Handle List types
    if origin is list:
        item_type = args[0] if args else str
        return [_generate_mock_value(item_type)]

    # Handle Dict types
    if origin is dict:
        return {}

    # Handle Enum types
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return list(field_type)[0].value

    # Handle basic types
    if field_type is str or field_type == str:
        return "mock_string"
    if field_type is int or field_type == int:
        return 42
    if field_type is float or field_type == float:
        return 3.14
    if field_type is bool or field_type == bool:
        return True

    # Handle Pydantic models
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        mock_data = {}
        for field_name, field_info in field_type.model_fields.items():
            mock_data[field_name] = _generate_mock_value(field_info.annotation)
        return field_type(**mock_data)

    # Default fallback
    return "mock_value"


def _create_mock_tool_call(tool: dict, call_id: str) -> ChatCompletionMessageToolCall:
    """Create a mock tool call response."""
    function_name = tool["function"]["name"]
    parameters = tool["function"].get("parameters", {})

    # Generate mock arguments based on the function schema
    mock_arguments = {}
    if "properties" in parameters:
        for prop_name, prop_schema in parameters["properties"].items():
            prop_type = prop_schema.get("type", "string")
            if prop_type == "string":
                if "enum" in prop_schema:
                    mock_arguments[prop_name] = prop_schema["enum"][0]
                else:
                    mock_arguments[prop_name] = f"mock_{prop_name}"
            elif prop_type == "integer":
                mock_arguments[prop_name] = 42
            elif prop_type == "number":
                mock_arguments[prop_name] = 3.14
            elif prop_type == "boolean":
                mock_arguments[prop_name] = True
            elif prop_type == "array":
                mock_arguments[prop_name] = []
            elif prop_type == "object":
                mock_arguments[prop_name] = {}

    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(
            name=function_name,
            arguments=json.dumps(mock_arguments),
        ),
    )


class MockAsyncOpenAI(AsyncOpenAI):
    """Mock AsyncOpenAI client that mimics the real client's interface."""

    def __init__(self, *args, **kwargs):
        self.base_url = kwargs.get("base_url", "https://mock.api.com/v1")
        self.api_key = kwargs.get("api_key", "mock_key")
        self.chat = MockChatCompletions()
        self.embeddings = MockEmbeddings()
        # Required for instructor compatibility check
        self.__class__.__module__ = "openai"
        self.__class__.__name__ = "AsyncOpenAI"


class MockChatCompletions:
    """Mock chat completions endpoint."""

    def __init__(self):
        self.completions = self

    async def create(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        stream: bool = False,
        tools: List[dict] | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a mock chat completion."""
        if stream:
            return self._create_stream(
                model, messages, tools, parallel_tool_calls, **kwargs
            )
        else:
            return self._create_non_stream(
                model, messages, tools, parallel_tool_calls, **kwargs
            )

    def _create_non_stream(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        tools: List[dict] | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a non-streaming mock completion."""
        completion_id = f"chatcmpl-mock-{int(time.time())}"
        timestamp = int(datetime.now().timestamp())

        # Generate response based on whether tools are provided
        if tools:
            # Generate tool calls
            tool_calls = []
            if parallel_tool_calls is False or parallel_tool_calls is None:
                # Only call the first tool
                tool_calls.append(_create_mock_tool_call(tools[0], f"call_mock_1"))
            else:
                # Call all tools
                for idx, tool in enumerate(tools):
                    tool_calls.append(
                        _create_mock_tool_call(tool, f"call_mock_{idx + 1}")
                    )

            message = ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=tool_calls,
            )
        else:
            # Generate standard content completion
            message = ChatCompletionMessage(
                role="assistant",
                content="This is a mock response from the MockModelClient.",
            )

        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop" if not tools else "tool_calls",
        )

        return ChatCompletion(
            id=completion_id,
            model=model,
            object="chat.completion",
            created=timestamp,
            choices=[choice],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )

    async def _create_stream(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        tools: List[dict] | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Create a streaming mock completion."""
        completion_id = f"chatcmpl-mock-{int(time.time())}"
        timestamp = int(datetime.now().timestamp())

        if tools:
            # Stream tool calls
            tool_calls = []
            if parallel_tool_calls is False or parallel_tool_calls is None:
                tool_calls.append(_create_mock_tool_call(tools[0], f"call_mock_1"))
            else:
                for idx, tool in enumerate(tools):
                    tool_calls.append(
                        _create_mock_tool_call(tool, f"call_mock_{idx + 1}")
                    )

            # Yield tool call chunks
            for tool_call in tool_calls:
                yield ChatCompletionChunk(
                    id=completion_id,
                    model=model,
                    object="chat.completion.chunk",
                    created=timestamp,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChoiceDelta(
                                role="assistant",
                                tool_calls=[tool_call],
                            ),
                            finish_reason=None,
                        )
                    ],
                )

            # Final chunk
            yield ChatCompletionChunk(
                id=completion_id,
                model=model,
                object="chat.completion.chunk",
                created=timestamp,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(),
                        finish_reason="tool_calls",
                    )
                ],
            )
        else:
            # Stream standard content
            content_chunks = ["This ", "is ", "a ", "mock ", "response."]
            for chunk_text in content_chunks:
                yield ChatCompletionChunk(
                    id=completion_id,
                    model=model,
                    object="chat.completion.chunk",
                    created=timestamp,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChoiceDelta(
                                role="assistant",
                                content=chunk_text,
                            ),
                            finish_reason=None,
                        )
                    ],
                )

            # Final chunk
            yield ChatCompletionChunk(
                id=completion_id,
                model=model,
                object="chat.completion.chunk",
                created=timestamp,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(),
                        finish_reason="stop",
                    )
                ],
            )


class MockEmbeddings:
    """Mock embeddings endpoint."""

    async def create(
        self,
        model: str,
        input: str | List[str],
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingModelResponse:
        """Create mock embeddings."""
        if isinstance(input, str):
            input = [input]

        dim = dimensions or 1536
        embeddings = []

        for idx, text in enumerate(input):
            embeddings.append(
                Embedding(
                    object="embedding",
                    index=idx,
                    embedding=[0.0] * dim,
                )
            )

        return EmbeddingModelResponse(
            object="list",
            data=embeddings,
            model=model,
            usage={"prompt_tokens": len(input) * 10, "total_tokens": len(input) * 10},
        )


class MockModelClient(ModelClient):
    """Mock model client for testing purposes.

    This client creates a facade over a mock AsyncOpenAI client that generates
    appropriate mock responses for chat completions, structured outputs, and embeddings.
    """

    @property
    def name(self) -> str:
        return "mock"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """Initialize the mock model client."""
        from .openai import OpenAIModelClient

        # Create a mock AsyncOpenAI client
        mock_openai_client = MockAsyncOpenAI(
            base_url=base_url or "https://mock.api.com/v1",
            api_key=api_key or "mock_key",
        )

        # Use the OpenAI client wrapper with our mock client
        self._openai_client = OpenAIModelClient(openai_client=mock_openai_client)

    @property
    def base_url(self) -> str:
        return self._openai_client.base_url

    @property
    def api_key(self) -> str:
        return self._openai_client.api_key

    async def chat_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Generate a mock chat completion."""
        return await self._openai_client.chat_completion(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )

    async def structured_output(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        response_model: Type[T],
        instructor_mode: Any | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Generate a mock structured output.

        This delegates to the OpenAI client's structured_output method,
        which will use instructor to parse the mock responses.
        """
        return await self._openai_client.structured_output(
            model=model,
            messages=messages,
            response_model=response_model,
            instructor_mode=instructor_mode,
            stream=stream,
            **kwargs,
        )

    async def embedding(
        self,
        model: str,
        input: str | List[str],
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingModelResponse:
        """Generate mock embeddings."""
        return await self._openai_client.embedding(
            model=model,
            input=input,
            dimensions=dimensions,
            **kwargs,
        )
