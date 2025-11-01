"""zyx.ai.models.adapters.litellm"""

from __future__ import annotations

# NOTE:
# avoids litellm async client warnings
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="coroutine 'close_litellm_async_clients' was never awaited",
)

import logging
from collections.abc import AsyncIterable, Iterable, Callable
from functools import lru_cache
from importlib.util import find_spec
from typing import Type, Tuple
import time

from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse

from ....core.exceptions import (
    LiteLLMModelAdapterError,
    ModelProviderInferenceException,
)
from ...utils.structured_outputs import (
    from_litellm,
    StructuredOutputType,
    AsyncInstructor,
    InstructorMode,
    InstructorModeName,
    prepare_structured_output_model,
)
from ..providers import ModelProviderInfo, ModelProvider, ModelProviderName
from . import ModelAdapter

__all__ = ["LiteLLMModelAdapter"]


_logger = logging.getLogger(__name__)


_LITELLM_INSTANCE = None
"""Singleton library level instance of the LiteLLM module."""


def _raise_litellm_not_available_error() -> None:
    raise LiteLLMModelAdapterError(
        "LiteLLM is required to use non-OpenAI or OpenAI-like model providers.\n"
        "You can install LiteLLM via pip with either:\n\n"
        "    `pip install litellm`\n\n"
        "    `pip install 'zyx[litellm]'`\n"
    )


@lru_cache(maxsize=1)
def is_litellm_available() -> bool:
    """Check if the `litellm` library is available."""
    return find_spec("litellm") is not None


def is_litellm_initialized() -> bool:
    """Check if the `litellm` library has been initialized."""
    global _LITELLM_INSTANCE
    return _LITELLM_INSTANCE is not None


@lru_cache(maxsize=1)
def get_litellm():
    """Retrieve the LiteLLM library singleton, raising an error if not available."""
    global _LITELLM_INSTANCE

    if not is_litellm_available():
        _raise_litellm_not_available_error()

    if _LITELLM_INSTANCE is None:
        start_time = time.time()
        import litellm

        litellm.drop_params = True
        litellm.modify_params = True
        _LITELLM_INSTANCE = litellm

        _logger.info(
            "Completed One Time Load of `LiteLLM` in %.2f seconds",
            time.time() - start_time,
        )

    return _LITELLM_INSTANCE


class LiteLLMModelAdapter(ModelAdapter[Callable, StructuredOutputType]):
    """Model adapter for LiteLLM-based model providers.

    LiteLLM acts as a universal adapter that can communicate with any LLM provider.
    Unlike OpenAIModelAdapter which uses an AsyncOpenAI client instance, this adapter
    uses LiteLLM's module-level functions (acompletion, aembedding) directly.

    NOTE: The `client` attribute for this adapter represents `litellm.acompletion`
    callable rather than a class instance.

    Model Name Format:
        LiteLLM expects provider-prefixed model names:
        - "openai/gpt-4"
        - "anthropic/claude-3-opus"
        - "bedrock/anthropic.claude-v2"
        - etc.
    """

    _provider: ModelProviderInfo
    _client: Callable
    _instructor_client: AsyncInstructor

    # NOTE:
    # only if cases explicitly given, this normally
    # wont be needed when using litellm
    _api_key: str | None = None

    def __init__(
        self,
        provider: ModelProviderName | ModelProviderInfo | None = None,
        *,
        api_key: str | None = None,
    ):
        """Initialize the LiteLLM model adapter.

        Args:
            provider: The model provider name or info. If `None`, attempts to
                infer the provider based on environment variables or defaults to
                'openai'.
            api_key: Optional API key for the model provider.
        """
        self._provider = None
        self._api_key = None
        self._client = None
        self._instructor_client = None

        if not is_litellm_available():
            _raise_litellm_not_available_error()

        if provider is None:
            self._provider = ModelProviderInfo(
                name="LITELLM_DEFAULT",
                supported_adapters=["litellm"],
            )
        else:
            if isinstance(provider, str):
                self._provider = ModelProvider.get_provider(provider)
            else:
                self._provider = provider

            if not "litellm" in self._provider.supported_adapters:
                raise ModelProviderInferenceException(
                    model="Unknown",
                    message=(
                        f"The specified provider '{self._provider.name}' "
                        "does not support the 'litellm' adapter."
                    ),
                )

        # set api key if provided
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "litellm"

    @property
    def provider(self) -> ModelProviderInfo:
        return self._provider

    @property
    def client(self) -> Callable:
        """Returns the litellm.acompletion function.

        Note: This is a function, not a client instance. LiteLLM uses
        module-level functions rather than client objects.
        """
        if self._client is None:
            # Lazy load LiteLLM and get the acompletion function
            litellm = get_litellm()
            self._client = litellm.acompletion
        return self._client

    @property
    def instructor_patch_fn(self) -> Callable:
        return from_litellm

    @property
    def instructor_mode(self) -> InstructorMode:
        if self._instructor_client is not None:
            return self._instructor_client.mode
        return None

    @property
    def instructor_client(self) -> AsyncInstructor:
        return self._instructor_client

    def get_instructor_client(
        self, instructor_mode: InstructorMode | str | None = None
    ) -> AsyncInstructor:
        """Retrieve an `Instructor` client patched from LiteLLM.

        Args:
            instructor_mode: The instructor mode to use. Defaults to TOOLS.

        Returns:
            AsyncInstructor instance configured for LiteLLM
        """
        if isinstance(instructor_mode, str):
            instructor_mode = InstructorMode(instructor_mode)

        if not self._instructor_client:
            if not instructor_mode:
                instructor_mode = InstructorMode.TOOLS

            # LiteLLM instructor client is created differently - it patches
            # the completion function, not a client instance
            self._instructor_client = self.instructor_patch_fn(
                self.client,  # This is litellm.acompletion
                mode=instructor_mode,
            )

        if instructor_mode and self._instructor_client.mode != instructor_mode:
            self._instructor_client.mode = instructor_mode

        return self._instructor_client

    async def create_chat_completion(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs,
    ) -> ChatCompletion | AsyncIterable[ChatCompletionChunk]:
        """Create a chat completion using LiteLLM.

        Args:
            model: Model name (should be provider-prefixed for LiteLLM,
                e.g., "openai/gpt-4", "anthropic/claude-3-opus")
            messages: Chat messages
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to LiteLLM

        Returns:
            ChatCompletion or async iterable of ChatCompletionChunk
        """

        # Ensure we have the client
        client = self.client

        # Add API key to kwargs if we have one
        if self._api_key:
            kwargs["api_key"] = self._api_key

        try:
            if stream:

                async def _stream_gen():
                    async for chunk in await client(
                        model=model, messages=messages, stream=True, **kwargs
                    ):
                        # Validate and convert LiteLLM response to OpenAI format
                        yield _validate_litellm_response(
                            chunk, "chat_completion:stream"
                        )

                return _stream_gen()
            else:
                response = await client(
                    model=model, messages=messages, stream=False, **kwargs
                )
                # Validate and convert LiteLLM response to OpenAI format
                return _validate_litellm_response(response, "chat_completion")

        except Exception as e:
            raise LiteLLMModelAdapterError(
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
        """Create a structured output using LiteLLM and Instructor.

        Args:
            model: Model name (provider-prefixed for LiteLLM)
            messages: Chat messages
            response_model: Pydantic model for structured output
            instructor_mode: Instructor mode to use
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Tuple of (structured_output, completion) or async iterable of tuples
        """
        # handle response model
        if not isinstance(response_model, BaseModel):
            response_model = prepare_structured_output_model(response_model)

        instructor_client = self.get_instructor_client(instructor_mode)

        completion: ChatCompletion | ChatCompletionChunk = None

        def _response_callback(response: ChatCompletion | ChatCompletionChunk):
            nonlocal completion
            if stream:
                completion = _validate_litellm_response(
                    response, "chat_completion:stream"
                )
            else:
                completion = _validate_litellm_response(response, "chat_completion")

        # Register callback to capture the raw completion
        instructor_client.on("completion:response", _response_callback)

        # Ensure instructor mode is set correctly
        if isinstance(instructor_mode, str):
            instructor_mode = InstructorMode(instructor_mode)

        if instructor_mode and instructor_client.mode != instructor_mode:
            instructor_client.mode = instructor_mode

        # Add API key to kwargs if we have one
        if self._api_key:
            kwargs["api_key"] = self._api_key

        if stream:

            async def _gen():
                try:
                    async for (
                        output
                    ) in instructor_client.chat.completions.create_partial(
                        model=model,
                        messages=messages,
                        response_model=response_model,
                        **kwargs,
                    ):
                        yield (output, completion)
                except Exception as e:
                    raise LiteLLMModelAdapterError(
                        f"Error during streamed structured output generation "
                        f"with model '{model}': {e}"
                    ) from e

            return _gen()
        else:
            try:
                output = await instructor_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_model=response_model,
                    **kwargs,
                )
            except Exception as e:
                raise LiteLLMModelAdapterError(
                    f"Error during structured output generation with model "
                    f"'{model}': {e}"
                ) from e
            return (output, completion)

    async def create_embedding(
        self, model: str, input: Iterable[str], **kwargs
    ) -> CreateEmbeddingResponse:
        """Create embeddings using LiteLLM.

        Args:
            model: Model name (provider-prefixed for LiteLLM,
                e.g., "openai/text-embedding-3-small")
            input: Text input(s) to embed
            **kwargs: Additional parameters

        Returns:
            CreateEmbeddingResponse
        """
        # Get LiteLLM module
        litellm = get_litellm()

        # Add API key to kwargs if we have one
        if self._api_key:
            kwargs["api_key"] = self._api_key

        try:
            response = await litellm.aembedding(model=model, input=input, **kwargs)
            return _validate_litellm_response(response, "embedding")
        except Exception as e:
            raise LiteLLMModelAdapterError(
                f"Error during embedding generation with model '{model}': {e}"
            ) from e


def _validate_litellm_response(
    response, response_type: str
) -> ChatCompletion | ChatCompletionChunk | CreateEmbeddingResponse:
    """Validate and convert LiteLLM responses to OpenAI format.

    LiteLLM returns its own response objects that are similar but not identical
    to OpenAI's. This function validates and converts them to proper OpenAI types.

    Args:
        response: LiteLLM response object
        response_type: Type of response ("chat_completion", "chat_completion:stream", "embedding")

    Returns:
        Validated OpenAI-format response

    Raises:
        ValueError: If response validation fails
    """
    try:
        if response_type == "chat_completion":
            return ChatCompletion.model_validate(response.model_dump())
        elif response_type == "chat_completion:stream":
            return ChatCompletionChunk.model_validate(response.model_dump())
        elif response_type == "embedding":
            return CreateEmbeddingResponse.model_validate(response.model_dump())
        else:
            raise ValueError(
                f"Unsupported response type '{response_type}' for LiteLLM response validation."
            )
    except Exception as e:
        raise ValueError(
            f"Failed to validate LiteLLM response of type '{response_type}': {e}"
        ) from e
