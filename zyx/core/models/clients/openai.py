"""zyx.models.clients.openai"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Type, TypeVar, overload

from httpx import AsyncClient
from instructor import AsyncInstructor, Mode, from_openai
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from openai.types.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from openai.types.create_embedding_response import (
    CreateEmbeddingResponse as EmbeddingModelResponse,
)
from openai.types.embedding_create_params import EmbeddingCreateParams

from ...._internal._exceptions import ModelClientError
from . import ModelClient, StructuredOutput

_logger = logging.getLogger(__name__)


T = TypeVar("T")


_CACHED_OPENAI_CLIENTS: Dict[str, AsyncOpenAI] = {}
"""Cached OpenAI clients by base URL."""


_SHARED_ASYNC_HTTPX_CLIENT: AsyncClient | None = None
"""Shared HTTPX client for all OpenAI clients."""


def shared_async_httpx_client() -> AsyncClient:
    """Get or create the shared HTTPX client."""
    global _SHARED_ASYNC_HTTPX_CLIENT

    if _SHARED_ASYNC_HTTPX_CLIENT is None:
        _SHARED_ASYNC_HTTPX_CLIENT = AsyncClient()

    if not _SHARED_ASYNC_HTTPX_CLIENT.is_closed:
        _SHARED_ASYNC_HTTPX_CLIENT = AsyncClient()

    return _SHARED_ASYNC_HTTPX_CLIENT


class OpenAIModelClient(ModelClient):
    """Model client for OpenAI compatible API providers."""

    @property
    def name(self) -> str:
        return "openai"

    @overload
    def __init__(
        self,
        base_url: str,
        api_key: str,
        http_client: AsyncClient | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        openai_client: AsyncOpenAI,
    ) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        http_client: AsyncClient | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        """Initialize the OpenAI model client with a Base URL and API key
        combination, or with a custom AsyncOpenAI client."""

        self._instructor_client: AsyncInstructor | None = None

        if openai_client:
            assert not (any([base_url, api_key, http_client])), (
                "Cannot provide base_url, api_key, or http_client when an OpenAI client is provided."
            )

            self._base_client = openai_client

            _logger.debug(
                f"OpenAIModelClient: Using provided OpenAI client for base URL {openai_client.base_url}."
            )

        if not base_url:
            base_url = "https://api.openai.com/v1"
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")

        else:
            global _CACHED_OPENAI_CLIENTS
            if base_url in _CACHED_OPENAI_CLIENTS:
                self._base_client = _CACHED_OPENAI_CLIENTS[base_url]

                _logger.debug(
                    f"OpenAIModelClient: Using cached OpenAI client for base URL {base_url}."
                )
            else:
                self._base_client = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    http_client=http_client or shared_async_httpx_client(),
                )

                _logger.debug(
                    f"OpenAIModelClient: Created new OpenAI client for base URL {base_url}."
                )

                _CACHED_OPENAI_CLIENTS[base_url] = self._base_client

    @property
    def base_url(self) -> str:
        return self._base_client.base_url

    @property
    def api_key(self) -> str:
        return self._base_client.api_key

    @property
    def base_client(self) -> AsyncOpenAI:
        return self._base_client

    def instructor_client(
        self, mode: Mode | None = None
    ) -> AsyncInstructor:
        """Get the instructor client for this model client. If no instructor client is
        cached, one will be created and cached. Optionally, include a specific generation
        / parsing mode to use from `instructor.mode.Mode`.
        """

        if self._instructor_client is None:
            self._instructor_client = from_openai(
                self._base_client,
            )

        if mode:
            if self._instructor_client.mode != mode:
                self._instructor_client = from_openai(
                    self._base_client,
                    mode=mode,
                )
        return self._instructor_client

    async def chat_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        # filter none values
        params = {k: v for k, v in params.items() if v is not None}

        params = (
            CompletionCreateParamsNonStreaming(**params)
            if not stream
            else CompletionCreateParamsStreaming(**params)
        )

        _logger.debug(
            f"OpenAIModelClient: Generating chat completion with model {model}."
            f"Stream: {stream}"
        )

        try:
            return await self._base_client.chat.completions.create(
                **params
            )
        except Exception as e:
            raise ModelClientError(
                f"Error generating chat completion: {e}",
                client=self.name,
                model=model,
            ) from e

    async def batch_chat_completion(
        self,
        model: str,
        messages: List[List[ChatCompletionMessageParam]],
        **kwargs: Any,
    ) -> List[ChatCompletion]:
        """Generate multiple chat completions concurrently.

        Args:
            model: The model to use for all completions
            messages: A list of message lists, one for each completion request
            **kwargs: Additional parameters to pass to each completion request

        Returns:
            A list of ChatCompletion objects in the same order as the input messages
        """
        _logger.debug(
            f"OpenAIModelClient: Generating {len(messages)} concurrent chat completions with model {model}."
        )

        # Create tasks for all completion requests
        tasks = [
            self.chat_completion(
                model=model, messages=msg_list, stream=False, **kwargs
            )
            for msg_list in messages
        ]

        try:
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions and raise if any occurred
            completions: List[ChatCompletion] = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    raise ModelClientError(
                        f"Error generating chat completion for request {idx}: {result}",
                        client=self.name,
                        model=model,
                    ) from result
                completions.append(result)

            return completions

        except Exception as e:
            if isinstance(e, ModelClientError):
                raise
            raise ModelClientError(
                f"Error generating batch chat completions: {e}",
                client=self.name,
                model=model,
            ) from e

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

        instructor_client = self.instructor_client(
            mode=instructor_mode,
        )

        raw_response: ChatCompletion | ChatCompletionChunk | None = None

        def _raw_response_callback(response):
            nonlocal raw_response
            raw_response = response

        instructor_client.on("completion:response", _raw_response_callback)

        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        params = (
            CompletionCreateParamsNonStreaming(**params)
            if not stream
            else CompletionCreateParamsStreaming(**params)
        )
        params["response_model"] = response_model

        # instructor context arg
        # this is rarely passed directly by zyx in this context, but we
        # do use instructor's templating utility directly before the
        # response is sent.

        # ctx in this case would only be passed by user if interacting
        # with this client directly.
        if kwargs.get("context", None) is not None:
            params["context"] = kwargs["context"]

            _logger.warning(
                f"Found `context` argument in kwargs during structured output generation, unless this was passed directly by the user, this"
                "is likely a bug in the calling code. Please raise an issue on GitHub is this warning comes up on"
                "its own."
            )

        async def _stream_closure():
            _logger.debug(
                f"OpenAIModelClient: Streaming structured output of type {response_model.__name__} with model {model} with instructor mode {instructor_client.mode}."
            )

            async for chunk in await instructor_client.create_partial(
                **params,
            ):
                try:
                    yield StructuredOutput(
                        output=chunk,
                        raw=raw_response,
                        instructor_mode=instructor_client.mode,
                    )
                except Exception as e:
                    raise ModelClientError(
                        f"Error generating structured output: {e}",
                        client=self.name,
                        model=model,
                    ) from e

        if stream:
            return await _stream_closure()

        else:
            _logger.debug(
                f"OpenAIModelClient: Generating structured output of type {response_model.__name__} with model {model} with instructor mode {instructor_client.mode}."
            )

            response = await instructor_client.create(
                **params,
            )

            try:
                return StructuredOutput(
                    output=response,
                    raw=raw_response,
                    instructor_mode=instructor_client.mode,
                )
            except Exception as e:
                raise ModelClientError(
                    f"Error generating structured output: {e}",
                    client=self.name,
                    model=model,
                ) from e

    async def batch_structured_output(
        self,
        model: str,
        messages: List[List[ChatCompletionMessageParam]],
        response_model: Type[T],
        instructor_mode: Mode | None = None,
        **kwargs: Any,
    ) -> List[StructuredOutput[T]]:
        """Generate multiple structured outputs concurrently.

        Args:
            model: The model to use for all structured outputs
            messages: A list of message lists, one for each structured output request
            response_model: The Pydantic model to parse responses into
            instructor_mode: Optional instructor mode to use
            **kwargs: Additional parameters to pass to each request

        Returns:
            A list of StructuredOutput objects in the same order as the input messages
        """
        _logger.debug(
            f"OpenAIModelClient: Generating {len(messages)} concurrent structured outputs of type {response_model.__name__} with model {model}."
        )

        # Create tasks for all structured output requests
        tasks = [
            self.structured_output(
                model=model,
                messages=msg_list,
                response_model=response_model,
                instructor_mode=instructor_mode,
                stream=False,
                **kwargs,
            )
            for msg_list in messages
        ]

        try:
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions and raise if any occurred
            outputs: List[StructuredOutput[T]] = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    raise ModelClientError(
                        f"Error generating structured output for request {idx}: {result}",
                        client=self.name,
                        model=model,
                    ) from result
                outputs.append(result)

            return outputs

        except Exception as e:
            if isinstance(e, ModelClientError):
                raise
            raise ModelClientError(
                f"Error generating batch structured outputs: {e}",
                client=self.name,
                model=model,
            ) from e

    async def embedding(
        self,
        model: str,
        input: str | List[str],
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingModelResponse:
        """Generate embeddings for a given input."""

        params = {
            "model": model,
            "input": input,
        }
        if dimensions:
            params["dimensions"] = dimensions

        params = EmbeddingCreateParams(**params)

        _logger.debug(
            f"""
            OpenAIModelClient: Generating embeddings for input of length {len(input)} using model {model} with dimensions {dimensions}.
            """
        )

        try:
            return await self._base_client.embeddings.create(**params)
        except Exception as e:
            raise ModelClientError(
                f"Error generating embeddings: {e}",
                client=self.name,
                model=model,
            ) from e
