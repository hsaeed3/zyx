"""zyx.models.adapters.openai"""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable
from functools import lru_cache
from typing import Type, Tuple, overload, Dict

from instructor import from_openai, AsyncInstructor, Mode as InstructorMode
from httpx import AsyncClient
from openai import AsyncOpenAI
from openai.types.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from openai.types.embedding_create_params import EmbeddingCreateParams
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse

from ...core._logging import _get_logger
from ..providers import (
    ModelProvider,
    ModelProviderName,
    MODEL_PROVIDERS,
    custom_model_provider,
)
from . import ModelAdapter, ResponseModel


_logger = _get_logger(__name__)


_OPENAI_MODEL_ADAPTER_ASYNC_HTTP_CLIENT: AsyncClient | None = None
"""Singlton instance of an `AsyncClient` used by default for all
`OpenAIModelAdapter` instances that do not provide their own `http_client`."""


_OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS: Dict[
    ModelProviderName | str, AsyncOpenAI
] = {}
"""Cache of `AsyncOpenAI` clients keyed by their provider name or custom
base url string."""


class OpenAIModelAdapter(ModelAdapter[AsyncOpenAI, ResponseModel]):
    """Model backend for all direct OpenAI API models, compatible OpenAI
    providers, and any custom defined providers through a base url.
    """

    _provider: ModelProvider

    _client: AsyncOpenAI
    _instructor_client: AsyncInstructor

    @overload
    def __init__(
        self,
        *,
        provider: ModelProvider | ModelProviderName | str = "openai",
        api_key: str | None = None,
        http_client: AsyncClient | None = None,
    ): ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI): ...

    def __init__(
        self,
        *,
        provider: ModelProvider | ModelProviderName | str | None = None,
        api_key: str | None = None,
        http_client: AsyncClient | None = None,
        openai_client: AsyncOpenAI | None = None,
    ):
        """Initialize a new `OpenAIModelAdapter` instance.

        Args:
            provider: The `ModelProvider`, `ModelProviderName` string, or custom
                base url string to use for this backend. If not provided,
                defaults to "openai".
            api_key: An optional API key string to use for authentication with
                the provider's API. If not provided, the backend will attempt to
                retrieve the API key from the environment or the provider's
                default method.
            http_client: An optional custom `AsyncClient` instance to use for
                making HTTP requests to the provider's API. If not provided,
                a singleton cached `AsyncClient` will be used.
            openai_client: An optional pre-initialized `AsyncOpenAI` client
                instance to use directly. If provided, the `provider`, `api_key`,
                and `http_client` parameters must not be provided.
        """
        self._client: AsyncOpenAI = None
        self._instructor_client: AsyncInstructor = None
        self._provider: ModelProvider = None

        assert provider is not None or openai_client is not None, (
            "Either a `provider` or an existing `openai_client` must be provided to instantiate an OpenAIModelAdapter.\n\n"
            "- You can define a provider using either a known provider name, a custom base url, or a pre-initialized `ModelProvider` instance.\n"
            "- Alternatively, you can provide an existing `AsyncOpenAI` client instance directly."
        )

        if openai_client is not None:
            assert provider is None, (
                "Cannot provide both an existing `openai_client` and a `provider` parameter."
            )
            assert api_key is None, (
                "Cannot provide both an existing `openai_client` and an `api_key` parameter."
            )
            assert http_client is None, (
                "Cannot provide both an existing `openai_client` and a custom `http_client` parameter."
            )

            self._client = openai_client
            self._provider = custom_model_provider(
                base_url=self._client.base_url, api_key=self._client.api_key
            )
        else:
            if isinstance(provider, ModelProvider):
                self._provider = provider

                self._client = AsyncOpenAI(
                    api_key=self._provider.get_api_key(api_key),
                    base_url=self._provider.base_url,
                    http_client=http_client or cached_async_http_client(),
                )

                _logger.debug(
                    f"Inherited preinitialized AsyncOpenAI client for provider or base url: {self._provider.name}."
                )
            else:
                if isinstance(provider, str):
                    # NOTE:
                    # if a custom provider has been initialized once, it should
                    # be able to pull it on the next initialization if needed
                    if provider.lower() in MODEL_PROVIDERS:
                        self._provider = MODEL_PROVIDERS[provider.lower()]

                        global _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS

                        if (
                            provider.lower()
                            in _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS
                        ):
                            self._client = (
                                _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS[
                                    provider.lower()
                                ]
                            )

                            _logger.debug(
                                f"Reusing cached AsyncOpenAI client for provider: {provider.lower()}."
                            )
                        else:
                            self._client = AsyncOpenAI(
                                api_key=self._provider.get_api_key(api_key),
                                base_url=self._provider.base_url,
                                http_client=http_client or cached_async_http_client(),
                            )
                            _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS[
                                provider.lower()
                            ] = self._client

                            _logger.debug(
                                f"Initialized a new AsyncOpenAI client for provider: {provider.lower()}."
                            )
                    else:
                        self._provider = custom_model_provider(
                            base_url=provider, api_key=api_key
                        )

                        self._client = AsyncOpenAI(
                            api_key=self._provider.get_api_key(api_key),
                            base_url=self._provider.base_url,
                            http_client=http_client or cached_async_http_client(),
                        )
                        _OPENAI_MODEL_ADAPTER_CACHED_ASYNC_OPENAI_CLIENTS[provider] = (
                            self._client
                        )

                        _logger.debug(
                            f"Initialized a new AsyncOpenAI client for custom base url: {provider}."
                        )
                else:
                    raise ValueError(
                        "The `provider` parameter must be either a `ModelProvider`, "
                        "`ModelProviderName` string, or a custom base url string."
                    )

    @property
    def name(self) -> str:
        return "openai"

    @property
    def provider(self) -> ModelProvider:
        return self._provider

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @property
    def instructor_patch_fn(self):
        return from_openai

    @property
    def instructor_mode(self) -> InstructorMode:
        if self._instructor_client is not None:
            return self._instructor_client.mode

    @property
    def instructor_client(self) -> AsyncInstructor:
        return self._instructor_client

    def get_instructor_client(
        self, instructor_mode: InstructorMode | str | None = None
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

    async def create_chat_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs,
    ) -> ChatCompletion | AsyncIterable[ChatCompletionChunk]:
        _logger.debug(
            f"Creating chat completion with model: {model}, stream: {stream}, using OpenAIModelAdapter."
        )
        _logger.debug(
            f"OpenAIModelAdapter generating chat completion for model '{model}' "
            f"with stream={stream}."
        )

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
            raise RuntimeError(
                f"Error during chat completion with model '{model}': {e}"
            ) from e

    async def create_structured_output(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        response_model: Type[ResponseModel],
        instructor_mode: InstructorMode | str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> (
        Tuple[ResponseModel, ChatCompletion]
        | AsyncIterable[Tuple[ResponseModel, ChatCompletionChunk]]
    ):
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
                _logger.debug(
                    f"OpenAIModelAdapter generating structured output stream "
                    f"for model '{model}' with response model '{response_model.__name__}'."
                )

                try:
                    async for output in client.chat.completions.create_partial(
                        model=model,
                        messages=messages,
                        response_model=response_model,
                        **kwargs,
                    ):
                        yield (output, completion)
                except Exception as e:
                    raise RuntimeError(
                        f"Error during streamed structured output generation "
                        f"with model '{model}': {e}"
                    ) from e

            return _gen()
        else:
            _logger.debug(
                f"OpenAIModelAdapter generating structured output "
                f"for model '{model}' with response model '{response_model.__name__}'."
            )

            try:
                output = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_model=response_model,
                    **kwargs,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error during structured output generation with model "
                    f"'{model}': {e}"
                ) from e

            return (output, completion)

    async def create_embedding(
        self, model: str, input: Iterable[str], **kwargs
    ) -> CreateEmbeddingResponse:
        _logger.debug(f"OpenAIModelAdapter generating embeddings for model '{model}'.")

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
            raise RuntimeError(
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
