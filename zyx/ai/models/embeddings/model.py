"""zyx.ai.models.embeddings.model"""

from __future__ import annotations

import logging
from typing import overload

from ..adapters import ModelAdapter, ModelAdapterName
from ..adapters.openai import OpenAIModelAdapter
from ..adapters.litellm import LiteLLMModelAdapter, is_litellm_initialized
from ..providers import (
    ModelProviderInfo,
    ModelProviderName,
    ModelProvider,
)
from .types import (
    EmbeddingModelName,
    EmbeddingModelResponse,
    EmbeddingModelSettings,
)

__all__ = ["EmbeddingModel", "embed", "aembed"]


_logger = logging.getLogger(__name__)


class EmbeddingModel:
    """An embedding model that can generate vector representations
    for a given input.
    """

    _model: str
    _adapter: ModelAdapter
    _settings: EmbeddingModelSettings | None

    @overload
    def __init__(
        self,
        model: EmbeddingModelName | str,
        *,
        settings: EmbeddingModelSettings | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: EmbeddingModelName | str,
        *,
        adapter: ModelAdapter | ModelAdapterName,
        settings: EmbeddingModelSettings | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: EmbeddingModelName | str,
        *,
        api_key: str | None = None,
        provider: ModelProviderInfo | ModelProviderName | str,
        settings: EmbeddingModelSettings | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: EmbeddingModelName | str,
        *,
        base_url: str,
        api_key: str | None = None,
        settings: EmbeddingModelSettings | None = None,
    ) -> None: ...

    def __init__(
        self,
        model: EmbeddingModelName | str = "openai/text-embedding-3-small",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        adapter: ModelAdapter | ModelAdapterName | None = None,
        provider: ModelProviderInfo | ModelProviderName | str | None = None,
        settings: EmbeddingModelSettings | None = None,
    ) -> None:
        """Initialize an EmbeddingModel instance.

        Args:
            model: The model name or ID to use for generating embeddings.
            base_url: Optional base URL for a custom model provider.
            api_key: Optional API key for authenticating with the model provider.
            adapter: Optional model adapter or adapter type to use.
            provider: Optional model provider or provider name to use.
            settings: Optional embedding model settings.
        """
        self._model = model
        self._settings = settings
        self._adapter = None

        assert not (base_url is not None and provider is not None), (
            "Cannot specify both 'base_url' and 'provider'. "
            "Please provide only one of these parameters."
        )

        if adapter is not None:
            if isinstance(adapter, ModelAdapter):
                # Use the provided adapter instance
                self._adapter = adapter
                _logger.debug(
                    f"Initialized EmbeddingModel with provided adapter instance: {adapter.name}"
                )
            else:
                self._adapter = self._create_adapter(
                    model=model,
                    provider=provider,
                    base_url=base_url,
                    api_key=api_key,
                    adapter_type=adapter,
                )
        else:
            # Auto-select adapter
            self._adapter = self._create_adapter(
                model=model,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                adapter_type="auto",
            )

    def _create_adapter(
        self,
        model: str,
        provider: ModelProvider | ModelProviderName | str | None,
        base_url: str | None,
        api_key: str | None,
        adapter_type: str,
    ) -> ModelAdapter:
        """Create the appropriate adapter for this model.

        Args:
            model: Model name
            provider: Provider configuration or None to infer
            base_url: Custom base URL for the API endpoint
            api_key: Optional API key
            adapter_type: Type of adapter to create

        Returns:
            Configured ModelAdapter instance
        """
        # Handle base_url case - create custom provider
        if base_url is not None:
            custom_provider = ModelProvider.custom_provider(
                base_url=base_url, api_key=api_key
            )
            if adapter_type == "litellm":
                return LiteLLMModelAdapter(provider=custom_provider, api_key=api_key)
            elif adapter_type == "openai":
                return OpenAIModelAdapter(provider=custom_provider, api_key=api_key)
            elif adapter_type == "auto":
                # For custom base URLs, prefer OpenAI adapter as it's more compatible
                _logger.debug(f"Using OpenAI adapter for custom base URL: {base_url}")
                return OpenAIModelAdapter(provider=custom_provider, api_key=api_key)
            else:
                raise ValueError(f"Unknown adapter_type: {adapter_type}")

        # Parse model string to extract provider if prefixed
        model_provider = None
        model_name = model

        if "/" in model:
            provider_prefix, model_name = model.split("/", 1)
            # If no explicit provider given, use the prefix
            if provider is None:
                provider = provider_prefix
                _logger.debug(f"Extracted provider '{provider}' from model string")

        # Infer provider if still not set
        if provider is None:
            inferred = ModelProvider.infer_model_provider(
                model=model,
                base_url=base_url,
                api_key=api_key,
                kind="embedding",
            )
            if inferred is None:
                raise ValueError(
                    f"Cannot infer provider from model name '{model}'. "
                    f"Please specify provider explicitly or use format 'provider/model'."
                )

            _logger.debug(f"Inferred provider: {inferred.name}")

            if inferred:
                provider = inferred

        # Determine adapter type
        if adapter_type == "litellm":
            # Force LiteLLM
            return LiteLLMModelAdapter(provider=provider, api_key=api_key)

        elif adapter_type == "openai":
            # Force OpenAI
            return OpenAIModelAdapter(provider=provider, api_key=api_key)

        elif adapter_type == "auto":
            # If LiteLLM has been initialized, prefer it for all providers
            if is_litellm_initialized():
                _logger.debug(
                    f"LiteLLM already initialized, using LiteLLM adapter for provider: {provider}"
                )
                return LiteLLMModelAdapter(provider=provider, api_key=api_key)

            # Auto-select based on provider compatibility
            # Get provider object if it's a string
            if isinstance(provider, str):
                provider_obj = ModelProvider.get_provider(provider)
            else:
                provider_obj = provider

            # Check if OpenAI-compatible
            # Providers that support OpenAI adapter: openai, openrouter, groq, etc.
            openai_compatible_providers = {
                "openai",
                "openrouter",
                "groq",
                "xai",
                "deepseek",
                "cerebras",
                "cohere",
                "fireworks",
                "github",
                "moonshotai",
                "ollama",
                "lm_studio",
            }

            provider_name = provider_obj.name if provider_obj else provider

            if provider_name in openai_compatible_providers or provider_name.startswith(
                "http"
            ):
                _logger.debug(f"Using OpenAI adapter for provider: {provider_name}")
                return OpenAIModelAdapter(provider=provider, api_key=api_key)
            else:
                _logger.debug(f"Using LiteLLM adapter for provider: {provider_name}")
                return LiteLLMModelAdapter(provider=provider, api_key=api_key)

        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

    @property
    def model(self) -> str:
        """The model name."""
        return self._adapter.provider.clean_model(self._model)

    @property
    def adapter(self) -> ModelAdapter:
        """The adapter used by this language model."""
        return self._adapter

    @property
    def settings(self) -> EmbeddingModelSettings | None:
        """Default settings for this embedding model."""
        return self._settings

    async def arun(
        self,
        input: str | list[str],
        dimensions: int | None = None,
        encoding_format: str | None = None,
        user: str | None = None,
        **kwargs,
    ) -> EmbeddingModelResponse:
        """Asynchronously generate embeddings for the given input.

        Args:
            input: Input text or list of texts to embed.
            dimensions: Optional number of dimensions for the output embeddings.
            encoding_format: Optional format for the embeddings ('float' or 'base64').
            user: Optional unique identifier representing the end-user.
            **kwargs: Additional parameters to pass to the adapter.

        Returns:
            EmbeddingModelResponse containing the generated embeddings.
        """
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input

        merged_settings = self._merge_settings(
            invocation_settings=EmbeddingModelSettings(
                dimensions=dimensions,
                encoding_format=encoding_format,
                user=user,
            ),
            kwargs=kwargs,
        )

        _logger.info(
            f"Generating embeddings using model '{self.model}' with adapter '{self._adapter.name}'"
        )

        return await self._adapter.create_embedding(
            model=self.model,
            input=inputs,
            **merged_settings,
            **kwargs,
        )

    def run(
        self,
        input: str | list[str],
        dimensions: int | None = None,
        encoding_format: str | None = None,
        user: str | None = None,
        **kwargs,
    ) -> EmbeddingModelResponse:
        """Synchronously generate embeddings for the given input.

        Args:
            input: Input text or list of texts to embed.
            dimensions: Optional number of dimensions for the output embeddings.
            encoding_format: Optional format for the embeddings ('float' or 'base64').
            user: Optional unique identifier representing the end-user.
            **kwargs: Additional parameters to pass to the adapter.

        Returns:
            EmbeddingModelResponse containing the generated embeddings.
        """
        import asyncio

        # Create a new event loop to avoid issues with closed or nested loops
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.arun(
                    input=input,
                    dimensions=dimensions,
                    encoding_format=encoding_format,
                    user=user,
                    **kwargs,
                )
            )
        finally:
            loop.close()

    def _merge_settings(
        self,
        invocation_settings: EmbeddingModelSettings | None,
        kwargs: dict,
    ) -> dict:
        """Merge default settings, invocation settings, and kwargs.

        Priority (highest to lowest):
        1. kwargs
        2. invocation_settings
        3. default settings (self._settings)

        Args:
            invocation_settings: Settings provided for this specific invocation
            kwargs: Additional keyword arguments

        Returns:
            Merged settings dictionary
        """
        merged = {}

        # Start with default settings
        if self._settings:
            merged.update(self._settings.model_dump(exclude_none=True))

        # Override with invocation settings
        if invocation_settings:
            merged.update(invocation_settings.model_dump(exclude_none=True))

        # Override with kwargs
        merged.update(kwargs)

        return merged


async def aembed(
    input: str | list[str],
    model: EmbeddingModelName | str = "openai/text-embedding-3-small",
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    adapter: ModelAdapter | ModelAdapterName | None = None,
    provider: ModelProviderInfo | ModelProviderName | str | None = None,
    dimensions: int | None = None,
    encoding_format: str | None = None,
    user: str | None = None,
    settings: EmbeddingModelSettings | None = None,
    **kwargs,
) -> EmbeddingModelResponse:
    """Asynchronously generate embeddings in one call.

    Args:
        input: Input text or list of texts to embed.
        model: Model name (e.g., "openai/text-embedding-3-small", "ollama/nomic-embed-text")
        base_url: Custom base URL for the API endpoint. Cannot be used with provider.
        api_key: Optional API key override
        adapter: Pre-configured ModelAdapter instance or adapter type string.
            If ModelAdapter instance provided, provider, api_key are ignored.
            If AdapterType string ("auto", "openai", "litellm"), used for adapter selection.
        provider: Provider configuration (ModelProvider instance, provider name,
            or custom base URL). If not provided, will be inferred from model name.
        dimensions: Optional number of dimensions for the output embeddings.
        encoding_format: Optional format for the embeddings ('float' or 'base64').
        user: Optional unique identifier representing the end-user.
        settings: Settings for this specific invocation (overrides default settings)
        **kwargs: Additional parameters passed to the underlying adapter

    Returns:
        EmbeddingModelResponse containing the generated embeddings
    """
    embedding_model = EmbeddingModel(
        model=model,
        base_url=base_url,
        api_key=api_key,
        adapter=adapter,
        provider=provider,
        settings=settings,
    )

    return await embedding_model.arun(
        input=input,
        dimensions=dimensions,
        encoding_format=encoding_format,
        user=user,
        **kwargs,
    )


def embed(
    input: str | list[str],
    model: EmbeddingModelName | str = "openai/text-embedding-3-small",
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    adapter: ModelAdapter | ModelAdapterName | None = None,
    provider: ModelProviderInfo | ModelProviderName | str | None = None,
    dimensions: int | None = None,
    encoding_format: str | None = None,
    user: str | None = None,
    settings: EmbeddingModelSettings | None = None,
    **kwargs,
) -> EmbeddingModelResponse:
    """Synchronously generate embeddings in one call.

    Args:
        input: Input text or list of texts to embed.
        model: Model name (e.g., "openai/text-embedding-3-small", "ollama/nomic-embed-text")
        base_url: Custom base URL for the API endpoint. Cannot be used with provider.
        api_key: Optional API key override
        adapter: Pre-configured ModelAdapter instance or adapter type string.
            If ModelAdapter instance provided, provider, api_key are ignored.
            If AdapterType string ("auto", "openai", "litellm"), used for adapter selection.
        provider: Provider configuration (ModelProvider instance, provider name,
            or custom base URL). If not provided, will be inferred from model name.
        dimensions: Optional number of dimensions for the output embeddings.
        encoding_format: Optional format for the embeddings ('float' or 'base64').
        user: Optional unique identifier representing the end-user.
        settings: Settings for this specific invocation (overrides default settings)
        **kwargs: Additional parameters passed to the underlying adapter

    Returns:
        EmbeddingModelResponse containing the generated embeddings
    """
    embedding_model = EmbeddingModel(
        model=model,
        base_url=base_url,
        api_key=api_key,
        adapter=adapter,
        provider=provider,
        settings=settings,
    )

    return embedding_model.run(
        input=input,
        dimensions=dimensions,
        encoding_format=encoding_format,
        user=user,
        **kwargs,
    )
