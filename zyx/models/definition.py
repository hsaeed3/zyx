"""zyx.models.definition"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass

from ..core.exceptions import ModelDefinitionError
from .clients import ModelClient
from .clients.openai import OpenAIModelClient
from .providers import ModelProvider, ModelProviderName, ModelProviderRegistry

__all__ = ["ModelDefinition", "ModelSettings"]


_logger = logging.getLogger(__name__)


@dataclass
class ModelSettings(ABC):
    """Base class for additional configuration options that can be
    applied to a model."""

    @property
    def kind(self) -> str:
        """Get the kind of model settings. This should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the 'kind' property")

    def __str__(self):
        from .._lib._beautification import _pretty_print_model_settings

        return _pretty_print_model_settings(self)

    def __rich__(self):
        from .._lib._beautification import _rich_pretty_print_model_settings

        return _rich_pretty_print_model_settings(self)


class ModelDefinition(ABC):
    """Abstract base class for an AI model. A model 'definition' is both a
    representation of a model and its settings, and is used to infer and create
    a compatible model client to use for when interacting with the
    model provider."""

    @property
    def kind(self) -> str:
        return "definition"

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> ModelClient:
        return self._client

    @property
    def provider(self) -> ModelProvider:
        return self._provider

    @property
    def settings(self) -> ModelSettings:
        return self._settings

    def __init__(
        self,
        model: str,
        *,
        provider: ModelProviderName | ModelProvider | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        self._model = model
        self._client: ModelClient | None = None
        self._provider: ModelProvider | None = None
        self._settings: ModelSettings | None = settings

        if base_url:
            assert not provider, "Cannot provide base_url and a provider"

            # try to get custom provider if exists, all custom providers
            # use an OpenAI client
            if ModelProviderRegistry().get(f"custom:{base_url}"):
                self._provider = ModelProviderRegistry().get(f"custom:{base_url}")
            else:
                self._provider = ModelProviderRegistry().register_custom(
                    base_url, api_key
                )

        if provider:
            if isinstance(provider, str):
                self._provider = ModelProviderRegistry().get(provider)

                if not self._provider:
                    raise ValueError(f"Unknown provider: {provider}")

            elif isinstance(provider, ModelProvider):
                self._provider = provider
            else:
                raise ValueError(f"Invalid provider: {provider}")

        else:
            self._provider = ModelProviderRegistry().infer_from_model_name(
                model=self._model,
                kind="language_model",
            )

            if not self._provider:
                # this will fallback to the LiteLLM client
                self._provider = ModelProvider(name="unknown")

    def get_client(self) -> ModelClient:
        """Get the inferred / associated model client for this language model.

        This requires the `_provider` attribute to be set as a ModelProvider
        object."""

        if not self._provider:
            raise ValueError("Cannot get client without a provider")

        if not self._client:
            # Check if using mock model
            if self._model.lower() == "mock" or self._model.lower().startswith("mock/"):
                from .clients.mock import MockModelClient

                self._client = MockModelClient(
                    base_url=(
                        self._provider.base_url if self._provider.base_url else None
                    ),
                    api_key=(
                        self._provider.get_api_key()
                        if hasattr(self._provider, "get_api_key")
                        else None
                    ),
                )
            elif "openai" in self._provider.supported_clients:
                self._client = OpenAIModelClient(
                    base_url=self._provider.base_url,
                    api_key=self._provider.get_api_key(),
                )
            else:
                from .clients.litellm import LiteLLMModelClient

                try:
                    self._client = LiteLLMModelClient(
                        base_url=self._provider.base_url,
                        api_key=self._provider.get_api_key(),
                    )
                except Exception as e:
                    raise ModelDefinitionError(
                        f"Error creating LiteLLM model client: {e}",
                        model=self._model,
                    ) from e

        return self._client

    def __str__(self):
        from .._lib._beautification import _pretty_print_model

        return _pretty_print_model(self, self.kind)

    def __rich__(self):
        from .._lib._beautification import _rich_pretty_print_model

        return _rich_pretty_print_model(self, self.kind)
