"""zyx.models.adapters.litellm"""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable, Callable
from functools import lru_cache
from importlib.util import find_spec
from typing import Type, Tuple
import time

from instructor import (
    from_litellm,
    AsyncInstructor,
    Mode as InstructorMode
)
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


_LITELLM_INSTANCE = None
"""Singleton library level instance of the LiteLLM module."""


def _raise_litellm_not_available_error() -> None:
    raise ImportError(
        "LiteLLM is required to use non-OpenAI or OpenAI-like model providers. \n"
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

        _logger.info("Completed One Time Load of `LiteLLM` in %.2f seconds", time.time() - start_time)

    return _LITELLM_INSTANCE


class LiteLLMModelAdapter(ModelAdapter[Callable, ResponseModel]):
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

    _provider: ModelProvider
    _api_key: str | None
    _client: Callable
    _instructor_client: AsyncInstructor

    def __init__(
        self,
        *,
        provider: ModelProvider | ModelProviderName | None = None,
        api_key: str | None = None,
    ):
        """Initialize a new `LiteLLMModelAdapter` instance.
        
        Args:
            provider: The `ModelProvider`, `ModelProviderName` string, or custom
                base url string to use. If not provided, uses a default provider
                that lets LiteLLM handle all inference.
            api_key: An optional API key string to use for authentication.
                If not provided, LiteLLM will attempt to retrieve the API key
                from environment variables based on the model name.
        """
        self._provider = None
        self._api_key = None
        self._client = None
        self._instructor_client = None

        # Verify LiteLLM is available
        if not is_litellm_available():
            _raise_litellm_not_available_error()

        # Handle provider initialization
        if provider is None:
            # Use default provider - LiteLLM will infer everything from model name
            self._provider = MODEL_PROVIDERS["LITELLM_BACKEND_DEFAULT"]
            _logger.debug("Initialized LiteLLMModelAdapter with default provider (LiteLLM inference)")
        
        elif isinstance(provider, ModelProvider):
            # Direct ModelProvider instance
            self._provider = provider
            _logger.debug(f"Initialized LiteLLMModelAdapter with provider: {provider.name}")
        
        elif isinstance(provider, str):
            # String provider name or custom base URL
            if provider.lower() in MODEL_PROVIDERS:
                # Known provider
                self._provider = MODEL_PROVIDERS[provider.lower()]
                _logger.debug(f"Initialized LiteLLMModelAdapter with known provider: {provider.lower()}")
            else:
                # Custom base URL - create custom provider
                self._provider = custom_model_provider(
                    base_url=provider,
                    api_key=api_key,
                )
                _logger.debug(f"Initialized LiteLLMModelAdapter with custom base url: {provider}")
        else:
            raise ValueError(
                "The `provider` parameter must be either a `ModelProvider`, "
                "`ModelProviderName` string, or a custom base url string."
            )

        # Store API key if provided (overrides provider's default)
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "litellm"

    @property
    def provider(self) -> ModelProvider:
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
        self,
        instructor_mode: InstructorMode | str | None = None
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
                mode=instructor_mode
            )
            _logger.debug(f"Created Instructor client for LiteLLM with mode: {instructor_mode}")

        if instructor_mode and self._instructor_client.mode != instructor_mode:
            self._instructor_client.mode = instructor_mode

        return self._instructor_client

    async def create_chat_completion(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs
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
        _logger.debug(
            f"Creating chat completion with model: {model}, stream: {stream}, "
            f"using LiteLLMModelAdapter."
        )

        # Ensure we have the client
        client = self.client

        # Add API key to kwargs if we have one
        if self._api_key:
            kwargs['api_key'] = self._api_key

        try:
            if stream:
                async def _stream_gen():
                    async for chunk in await client(
                        model=model,
                        messages=messages,
                        stream=True,
                        **kwargs
                    ):
                        # Validate and convert LiteLLM response to OpenAI format
                        yield _validate_litellm_response(chunk, "chat_completion:stream")
                
                return _stream_gen()
            else:
                response = await client(
                    model=model,
                    messages=messages,
                    stream=False,
                    **kwargs
                )
                # Validate and convert LiteLLM response to OpenAI format
                return _validate_litellm_response(response, "chat_completion")

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
        **kwargs
    ) -> Tuple[ResponseModel, ChatCompletion] | AsyncIterable[Tuple[ResponseModel, ChatCompletionChunk]]:
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
        instructor_client = self.get_instructor_client(instructor_mode)

        completion: ChatCompletion | ChatCompletionChunk = None

        def _response_callback(response: ChatCompletion | ChatCompletionChunk):
            nonlocal completion
            if stream:
                completion = _validate_litellm_response(response, "chat_completion:stream")
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
            kwargs['api_key'] = self._api_key

        if stream:
            _logger.debug(
                f"LiteLLMModelAdapter generating structured output stream "
                f"for model '{model}' with response model '{response_model.__name__}'."
            )

            async def _gen():
                try:
                    async for output in instructor_client.chat.completions.create_partial(
                        model=model,
                        messages=messages,
                        response_model=response_model,
                        **kwargs
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
                f"LiteLLMModelAdapter generating structured output "
                f"for model '{model}' with response model '{response_model.__name__}'."
            )

            try:
                output = await instructor_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_model=response_model,
                    **kwargs
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error during structured output generation with model "
                    f"'{model}': {e}"
                ) from e

            return (output, completion)

    async def create_embedding(
        self,
        model: str,
        input: Iterable[str],
        **kwargs
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
        _logger.debug(
            f"LiteLLMModelAdapter generating embeddings for model '{model}'."
        )

        # Get LiteLLM module
        litellm = get_litellm()

        # Add API key to kwargs if we have one
        if self._api_key:
            kwargs['api_key'] = self._api_key

        try:
            response = await litellm.aembedding(
                model=model,
                input=input,
                **kwargs
            )
            return _validate_litellm_response(response, "embedding")
        except Exception as e:
            raise RuntimeError(
                f"Error during embedding generation with model '{model}': {e}"
            ) from e


def _validate_litellm_response(
    response,
    response_type: str
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