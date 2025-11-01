"""zyx.ai.models.adapters.openai"""

import logging
from collections.abc import AsyncIterable, Iterable
from functools import lru_cache
from typing import Type, Tuple, overload, Dict

from httpx import AsyncClient
from openai import AsyncOpenAI
from pydantic import BaseModel

from openai.types.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from openai.types.embedding_create_params import EmbeddingCreateParams
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse

from ....core.exceptions import ModelProviderInferenceException, OpenAIModelAdapterError
from ...utils.structured_outputs import (
    from_openai,
    StructuredOutputType,
    AsyncInstructor,
    InstructorMode,
    InstructorModeName,
    prepare_structured_output_model,
)
from ..providers import ModelProviderInfo, ModelProvider
from ..providers.openai import OpenAIModelProviderName
from . import ModelAdapter

__all__ = ["OpenAIModelAdapter"]


_OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT: AsyncClient | None = None
"""Singlton instance of an `AsyncClient` used by default for all
`OpenAIModelAdapter` instances that do not provide their own `http_client`."""


_OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS: Dict[
    OpenAIModelProviderName | str, AsyncOpenAI
] = {}
"""Cache of `AsyncOpenAI` clients keyed by their provider name or custom
base url string."""


class OpenAIModelAdapter(ModelAdapter[AsyncOpenAI, StructuredOutputType]):
    """Model backend for all direct OpenAI API models, compatible OpenAI
    providers, and any custom defined providers through a base url.
    """

    _provider: ModelProviderInfo
    """The associated ModelProviderInfo for this adapter used to define
    it's client."""

    _client: AsyncOpenAI
    """The OpenAI Async client used to interact with the OpenAI or
    compatible API."""

    _instructor_client: AsyncInstructor
    """The AsyncInstructor client used to handle structured outputs
    through the `instructor` library."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def provider(self) -> ModelProviderInfo:
        return self._provider

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @property
    def instructor_patch_fn(
        self,
    ):
        return from_openai

    @property
    def instructor_mode(self) -> InstructorMode:
        if self._instructor_client:
            return self._instructor_client.mode
        return None

    @property
    def instructor_client(self) -> AsyncInstructor:
        return self._instructor_client

    def get_instructor_client(
        self, instructor_mode: InstructorMode | InstructorModeName | None = None
    ) -> AsyncInstructor:
        """Retrieve an `Instructor` client patched from the primary client associated
        with this model backend.

        If no `instructor_mode` is provided, the default mode set is
        `instructor.Mode.TOOLS`.
        """
        if isinstance(instructor_mode, str):
            instructor_mode = InstructorMode(instructor_mode)

        if not self._instructor_client:
            if not instructor_mode:
                instructor_mode = InstructorMode.TOOLS

            self._instructor_client = self.instructor_patch_fn(
                self.client, instructor_mode
            )

        if instructor_mode:
            self._instructor_client.mode = instructor_mode
        return self._instructor_client

    @overload
    def __init__(
        self,
        provider: OpenAIModelProviderName | str,
        *,
        api_key: str | None = None,
        http_client: AsyncClient | None = None,
    ): ...

    @overload
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        http_client: AsyncClient | None = None,
    ): ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI): ...

    def __init__(
        self,
        provider: ModelProviderInfo | OpenAIModelProviderName | str | None = None,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        http_client: AsyncClient | None = None,
        openai_client: AsyncOpenAI | None = None,
    ):
        global _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS
        global _OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT

        self._client: AsyncOpenAI = None
        self._instructor_client: AsyncInstructor = None
        self._provider: ModelProviderInfo = None

        assert (
            provider is not None or base_url is not None or openai_client is not None
        ), (
            "Either 'provider', 'base_url', or 'openai_client' must be provided to initialize an OpenAIModelAdapter."
        )

        if openai_client is not None:
            assert provider is None, (
                "If 'openai_client' is provided, 'provider' must be None."
            )
            assert base_url is None, (
                "If 'openai_client' is provided, 'base_url' must be None."
            )
            assert api_key is None, (
                "If 'openai_client' is provided, 'api_key' must be None."
            )
            assert http_client is None, (
                "If 'openai_client' is provided, 'http_client' must be None."
            )

            self._client = openai_client

            self._provider = ModelProvider.custom_provider(
                base_url=openai_client.base_url, api_key=openai_client.api_key
            )

        else:
            # handle base URL before provider
            if base_url is not None:
                assert provider is None, (
                    "If 'base_url' is provided, 'provider' must be None."
                )

                # creates or retrieved cached model provider
                self._provider = ModelProvider.custom_provider(
                    base_url=base_url, api_key=api_key
                )

                if (
                    self._provider.name
                    in _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS
                ):
                    self._client = _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS[
                        self._provider.name
                    ]
                else:
                    self._client = AsyncOpenAI(
                        api_key=self._provider.get_api_key(api_key),
                        base_url=self._provider.base_url,
                        http_client=http_client,
                    )
                    _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS[
                        self._provider.name
                    ] = self._client

            else:
                if isinstance(provider, str):
                    if provider in ModelProvider.list():
                        self._provider = ModelProvider.get_provider(provider)

                        if not "openai" in self._provider.supported_adapters:
                            raise ModelProviderInferenceException(
                                model="",
                                base_url=self._provider.base_url,
                                message=(
                                    f"The inferred ModelProviderInfo '{self._provider.name}' is not "
                                    "compatible with the OpenAIModelAdapter."
                                ),
                            )

                        if (
                            provider
                            in _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS
                        ):
                            self._client = (
                                _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS[
                                    provider
                                ]
                            )
                        else:
                            self._client = AsyncOpenAI(
                                api_key=self._provider.get_api_key(api_key),
                                base_url=self._provider.base_url,
                                http_client=http_client,
                            )
                            _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS[
                                provider
                            ] = self._client

                elif isinstance(provider, ModelProviderInfo):
                    self._provider = provider

                    if not "openai" in self._provider.supported_adapters:
                        raise ModelProviderInferenceException(
                            model="",
                            base_url=self._provider.base_url,
                            message=(
                                f"The provided ModelProviderInfo '{self._provider.name}' is not "
                                "compatible with the OpenAIModelAdapter."
                            ),
                        )

                    if (
                        self._provider.name
                        in _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS
                    ):
                        self._client = (
                            _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS[
                                self._provider.name
                            ]
                        )
                    else:
                        self._client = AsyncOpenAI(
                            api_key=self._provider.get_api_key(api_key),
                            base_url=self._provider.base_url,
                            http_client=http_client,
                        )
                        _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS[
                            self._provider.name
                        ] = self._client
                else:
                    raise ModelProviderInferenceException(
                        model="",
                        base_url=None,
                        message=(
                            "Failed to infer or configure a valid ModelProviderInfo for the "
                            "OpenAIModelAdapter with the provided parameters."
                        ),
                    )

    async def create_chat_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs,
    ) -> ChatCompletion | AsyncIterable[ChatCompletionChunk]:
        async def _stream_gen(params: dict):
            async for chunk in self.client.chat.completions.create(**params):
                yield chunk

        params = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        if stream:
            allowed_keys = CompletionCreateParamsStreaming.__annotations__.keys()
            for k, v in CompletionCreateParamsStreaming(**params).items():
                if k in allowed_keys:
                    params[k] = v
        else:
            allowed_keys = CompletionCreateParamsNonStreaming.__annotations__.keys()
            for k, v in CompletionCreateParamsNonStreaming(**params).items():
                if k in allowed_keys:
                    params[k] = v

        try:
            if stream:
                return _stream_gen(params)
            else:
                return await self.client.chat.completions.create(**params)
        except Exception as e:
            raise OpenAIModelAdapterError(
                f"Error during chat completion with model '{model}': {e}"
            ) from e

    async def create_structured_output(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        response_model: Type[StructuredOutputType],
        instructor_mode: InstructorMode | InstructorModeName | None = None,
        stream: bool = False,
        **kwargs,
    ) -> (
        Tuple[StructuredOutputType, ChatCompletion]
        | AsyncIterable[Tuple[StructuredOutputType, ChatCompletionChunk]]
    ):
        # process response model
        if not isinstance(response_model, BaseModel):
            response_model = prepare_structured_output_model(response_model)

        client = self.get_instructor_client(instructor_mode)

        completion: ChatCompletion | ChatCompletionChunk = None

        def _response_callback(response):
            nonlocal completion
            completion = response

        client.on("completion:response", _response_callback)

        if isinstance(instructor_mode, str):
            instructor_mode = InstructorMode(instructor_mode)

        if instructor_mode and client.mode != instructor_mode:
            client.mode = instructor_mode

        if stream:

            async def _gen():
                try:
                    async for output in client.chat.completions.create_partial(
                        model=model,
                        messages=messages,
                        response_model=response_model,
                        **kwargs,
                    ):
                        yield (output, completion)
                except Exception as e:
                    raise OpenAIModelAdapterError(
                        f"Error during streamed structured output generation "
                        f"with model '{model}': {e}"
                    ) from e

            return _gen()
        else:
            try:
                output = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_model=response_model,
                    **kwargs,
                )
            except Exception as e:
                raise OpenAIModelAdapterError(
                    f"Error during structured output generation with model "
                    f"'{model}': {e}"
                ) from e
            return (output, completion)

    async def create_embedding(
        self, model: str, input: Iterable[str], **kwargs
    ) -> CreateEmbeddingResponse:
        params = {
            "model": model,
            "input": input,
            **kwargs,
        }

        allowed_keys = EmbeddingCreateParams.__annotations__.keys()
        for k, v in EmbeddingCreateParams(**params).items():
            if k in allowed_keys:
                params[k] = v

        try:
            return await self.client.embeddings.create(**params)
        except Exception as e:
            raise OpenAIModelAdapterError(
                f"Error during embedding generation with model '{model}': {e}"
            ) from e


@lru_cache()
def cached_async_http_client() -> AsyncClient:
    global _OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT

    if _OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT is None:
        _OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT = AsyncClient()
        return _OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT

    if _OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT.is_closed:
        _OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT = AsyncClient()
    return _OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT
