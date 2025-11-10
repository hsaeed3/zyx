"""zyx.core.models.clients"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Generic, List, Type, TypedDict, TypeVar

from instructor import Mode
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from openai.types.create_embedding_response import (
    CreateEmbeddingResponse as EmbeddingModelResponse,
)

T = TypeVar("T")


class StructuredOutput(TypedDict, Generic[T]):
    """Raw representation of a structured output generated using
    the `instructor` library. We collect both the raw completion
    and parsed output in the response."""

    output: T
    raw: ChatCompletion | ChatCompletionChunk
    instructor_mode: Mode | None


class ModelClient(ABC):
    """Base class for a model client. A model client is used to interact and generate
    responses from API providers.

    A model client is 'dumb' and is only responsible for the interaction with a backend client
    object, no provider-specific logic is implemented here.
    """

    @property
    def name(self) -> str:
        """The name of the model client."""
        raise NotImplementedError()

    @abstractmethod
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize this model client with an optional base URL and API key. If no base URL is provided,
        this client will use the default base URL for the provider. (OpenAI, Anthropic, etc.)
        """
        raise NotImplementedError()

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Generate a chat completion, or stream chat completion chunks."""

    @abstractmethod
    async def structured_output(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        response_model: Type[T],
        instructor_mode: Mode | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> StructuredOutput[T] | AsyncIterator[StructuredOutput[T]]:
        """Generate a structured output from a model."""

    @abstractmethod
    async def embedding(
        self,
        model: str,
        input: str | List[str],
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingModelResponse:
        """Generate embeddings for a given input."""
