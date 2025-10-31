"""zyx.models.adapters"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Iterable, Callable
from typing import Generic, TypeVar, Type, Tuple

from instructor import (
    AsyncInstructor,
    Mode as InstructorMode
)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse

from ..providers import ModelProvider

__all__ = [
    "ModelAdapter",
    "ModelAdapterClient",
    "ResponseModel"
]


ModelAdapterClient = TypeVar("ModelAdapterClient")
"""Generic variable alias for the client object attached to a model adapter."""


ResponseModel = TypeVar("ResponseModel")
"""Helper typevar representation for the `response_model` parameter used when
generating structured outputs through `instructor`."""


class ModelAdapter(ABC, Generic[ModelAdapterClient, ResponseModel]):
    """Abstract base class for a model adapter. A model adapter is responsible
    for providing an interface for a specific LLM provider. A model adapter
    cannot be initialized without a valid `ModelProvider`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this model backend. (e.g., "openai", "litellm", etc.)"""
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def provider(self) -> ModelProvider:
        """The currently set `ModelProvider` associated with this model backend."""
    
    @property
    @abstractmethod
    def client(self) -> ModelAdapterClient:
        """The client object used to interact with the LLM provider's API.
        This is directly instantiated by the model adapter."""

    @property
    @abstractmethod
    def instructor_patch_fn(self) -> Callable[
        [ModelAdapterClient | Callable, InstructorMode], AsyncInstructor
    ]:
        """A callable that patches the primary client to return an
        `instructor.AsyncInstructor` client when called with an
        `InstructorMode`.
        
        (e.g., `instructor.from_openai()`, `instructor.from_litellm()`, etc.)
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def instructor_client(self) -> AsyncInstructor:
        """The default `instructor.AsyncInstructor` client patched from the
        primary client associated with this model backend. The default mode
        set is `instructor.Mode.TOOLS`.
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def instructor_mode(self) -> InstructorMode:
        """The currently set `instructor.Mode` if this model backend has
        instantiated an `instructor.AsyncInstructor` client.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_instructor_client(
        self,
        instructor_mode : InstructorMode | str | None = None
    ):
        """Retrieve an `Instructor` client patched from the primary client associated
        with this model backend.
        
        If no `instructor_mode` is provided, the default mode set is
        `instructor.Mode.TOOLS`.
        """
        raise NotImplementedError()
    
    @abstractmethod
    async def create_chat_completion(
        self,
        model : str,
        messages : Iterable[ChatCompletionMessageParam],
        stream : bool = False,
        **kwargs
    ) -> ChatCompletion | AsyncIterable[ChatCompletionChunk]:
        """Create a chat completion using the specified model and messages.

        Args:
            model (str): The model to use for the chat completion.
            messages (Iterable[ChatCompletionMessageParam]): The messages to
                include in the chat completion.
            stream (bool, optional): Whether to stream the response. Defaults
                to False.
            **kwargs: Additional keyword arguments to pass to the underlying
                API call.
        
        Returns:
            ChatCompletion | AsyncIterable[ChatCompletionChunk]: The chat
                completion response or an async iterable of chat completion
                chunks if streaming is enabled.
        """

    @abstractmethod
    async def create_structured_output(
        self,
        model : str,
        messages : Iterable[ChatCompletionMessageParam],
        response_model : Type[ResponseModel],
        instructor_mode : InstructorMode | str | None = None,
        stream : bool = False,
        **kwargs
    ) -> Tuple[ResponseModel, ChatCompletion] | AsyncIterable[Tuple[ResponseModel, ChatCompletionChunk]]:
        """Create a structured output using the specified model, messages,
        and response model.

        Args:
            model (str): The model to use for the structured output.
            messages (Iterable[ChatCompletionMessageParam]): The messages to
                include in the structured output.
            response_model (Type[ResponseModel]): The response model to use
                for structuring the output.
            instructor_mode (InstructorMode | str | None, optional): The
                instructor mode to use. Defaults to None.
            stream (bool, optional): Whether to stream the response. Defaults
                to False.
            **kwargs: Additional keyword arguments to pass to the underlying
                API call.
        
        Returns:
            Tuple[ResponseModel, ChatCompletion] | AsyncIterable[Tuple[ResponseModel, ChatCompletionChunk]]: 
                The structured output and chat completion
                response or an async iterable of structured outputs and chat
                completion chunks if streaming is enabled.
        """
        raise NotImplementedError()
    
    @abstractmethod
    async def create_embedding(
        self,
        model : str,
        input : list[str],
        **kwargs
    ) -> CreateEmbeddingResponse:
        """Create an embedding using the specified model and input.

        Args:
            model (str): The model to use for the embedding.
            input (list[str]): The input text to embed.
            **kwargs: Additional keyword arguments to pass to the
                underlying API call.

        Returns:
            CreateEmbeddingResponse: The embedding response.
        """
        raise NotImplementedError()