"""zyx.core.models.clients.litellm"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from functools import lru_cache
from importlib.util import find_spec
from typing import Any, List, Type, TypeVar

from instructor import AsyncInstructor, Mode, from_litellm
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from openai.types.create_embedding_response import (
    CreateEmbeddingResponse as EmbeddingModelResponse,
)
from pydantic import BaseModel

from ...._internal._exceptions import ModelClientError
from . import ModelClient, StructuredOutput

_logger = logging.getLogger(__name__)


T = TypeVar("T")
"""Type variable for the model client."""


class LiteLLMHelper:
    """Helper class used internally within for dependency management
    regarding the LiteLLM library."""

    @staticmethod
    @lru_cache(maxsize=1)
    def is_available() -> bool:
        """Check if LiteLLM package is installed.

        Returns:
            True if litellm can be imported
        """
        return find_spec("litellm") is not None

    @classmethod
    def get(cls):
        """Get or initialize the LiteLLM package instance.

        Returns:
            The litellm module

        Raises:
            ImportError: If litellm is not installed
        """
        if not cls.is_available():
            raise ImportError(
                "LiteLLM is required for non-OpenAI compatible providers.\n\n"
                "Install via pip:\n"
                "  pip install litellm\n"
                "  pip install 'zyx[litellm]'\n"
            )

        if not cls._initialized:
            import litellm

            litellm.drop_params = True
            litellm.modify_params = True
            cls._instance = litellm
            cls._initialized = True

            _logger.debug("Initialized LiteLLM library")

        return cls._instance


class LiteLLMModelClient(ModelClient):
    """Model client for the LiteLLM library.

    NOTE: This is used as the automatic fallback client for all 'unknown' models when
    the LiteLLM library is installed.
    """

    @property
    def name(self) -> str:
        return "litellm"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the LiteLLM model client."""

        self._instructor_client: AsyncInstructor | None = None

        # NOTE: never
        self._base_client = None

        self._base_url = base_url
        self._api_key = api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    def instructor_client(
        self, mode: Mode | None = None
    ) -> AsyncInstructor:
        """Get the instructor client for this model client. If no instructor client is
        cached, one will be created and cached. Optionally, include a specific generation
        / parsing mode to use from `instructor.mode.Mode`."""

        if not self._instructor_client:
            self._instructor_client = from_litellm(
                LiteLLMHelper.get().acompletion,
            )

        if mode:
            if self._instructor_client.mode != mode:
                self._instructor_client = from_litellm(
                    LiteLLMHelper.get().acompletion,
                    mode=mode,
                )
        return self._instructor_client

    def parse_litellm_response(
        self,
        response: BaseModel,
        stream: bool = False,
    ) -> ChatCompletion | ChatCompletionChunk:
        if stream:
            return ChatCompletionChunk.model_validate(
                response.model_dump(),
            )
        else:
            return ChatCompletion.model_validate(
                response.model_dump(),
            )

    async def chat_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs: Any,
    ):
        """Generate a chat completion, or stream chat completion chunks."""

        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        if self.base_url:
            params["base_url"] = self.base_url
        if self.api_key:
            params["api_key"] = self.api_key

        if stream:

            async def _stream_wrapper():
                _logger.debug(
                    f"LiteLLMModelClient: Streaming chat completion with model {model}."
                )

                try:
                    async for (
                        chunk
                    ) in await LiteLLMHelper.get().acompletion(
                        stream=True, **params
                    ):
                        yield self.parse_litellm_response(
                            chunk,
                            stream=True,
                        )
                except Exception as e:
                    raise ModelClientError(
                        f"Error generating chat completion: {e}",
                        client=self.name,
                        model=model,
                    ) from e

            return await _stream_wrapper()

        else:
            _logger.debug(
                f"LiteLLMModelClient: Generating chat completion with model {model}."
            )

            try:
                return self.parse_litellm_response(
                    await LiteLLMHelper.get().acompletion(**params),
                    stream=False,
                )
            except Exception as e:
                raise ModelClientError(
                    f"Error generating chat completion: {e}",
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
        params["response_model"] = response_model

        if self.base_url:
            params["base_url"] = self.base_url
        if self.api_key:
            params["api_key"] = self.api_key

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
                f"LiteLLMModelClient: Streaming structured output of type {response_model.__name__} with model {model} with instructor mode {instructor_client.mode}."
            )

            try:
                async for chunk in await instructor_client.create_partial(
                    **params,
                ):
                    yield StructuredOutput(
                        output=chunk,
                        raw=self.parse_litellm_response(
                            raw_response,
                            stream=True,
                        ),
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
                f"LiteLLMModelClient: Generating structured output of type {response_model.__name__} with model {model} with instructor mode {instructor_client.mode}."
            )

            response = await instructor_client.create(
                **params,
            )

            try:
                return StructuredOutput(
                    output=response,
                    raw=self.parse_litellm_response(
                        raw_response,
                        stream=False,
                    ),
                    instructor_mode=instructor_client.mode,
                )
            except Exception as e:
                raise ModelClientError(
                    f"Error generating structured output: {e}",
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
            f"LiteLLMModelClient: Generating {len(messages)} concurrent chat completions with model {model}."
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
            f"LiteLLMModelClient: Generating {len(messages)} concurrent structured outputs of type {response_model.__name__} with model {model}."
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

        if self.base_url:
            params["base_url"] = self.base_url
        if self.api_key:
            params["api_key"] = self.api_key

        _logger.debug(
            f"LiteLLMModelClient: Generating embeddings for input of length {len(input)} using model {model} with dimensions {dimensions}."
        )

        return await LiteLLMHelper.get().aembedding(
            **params,
        )
