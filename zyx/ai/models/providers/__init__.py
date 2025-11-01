"""zyx.ai.models.providers"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Literal, TypeAliasType, Dict, List, Tuple

from ....core.exceptions import ModelProviderException, ModelProviderInferenceException

__all__ = [
    "ModelProviderInfo",
    "ModelProviderName",
    "ModelProvider",
    "MODEL_PROVIDERS_REGISTRY",
]


MODEL_PROVIDERS_REGISTRY: Dict[str | ModelProviderName, ModelProviderInfo] = {}
"""A mapping of AI model provider names to their corresponding `AIProvider`. This is used both to lookup
known API providers, along with adding custom definitions with a 
custom base url."""


ModelProviderName = TypeAliasType(
    "ModelProviderName",
    Literal[
        # --- anthropic compat (anthropic is an extra)
        "anthropic",
        # --- openai compat
        "cohere",
        "cerebras",
        "deepseek",
        "fireworks",
        "github",
        "groq",
        "huggingfacelm_studio",
        "moonshotai",
        "ollama",
        "openai",
        "openrouter",
        "xai",
    ],
)
"""Combined alias of all supported AI model API provider
prefixes or names."""


@dataclass(frozen=True)
class ModelProviderInfo:
    """A representation of a set of configuration parameters and metadata for
    a specific AI model API provider (e.g. OpenAI, Anthropic, etc.).

    These are used both to infer broad model names along with models defined with explicit
    provider names to an appopriate set of configuration parameters as well as
    a compatible `adapter` (client) class for making requests to the provider's API.
    """

    name: str
    """The name of this model provider. (E.g. "openai", "anthropic", etc.)"""

    base_url: str | None = None
    """The base URL to use when initializing an adapter for this
    client's API."""

    api_key_env: str | None = None
    """The environment variable from which to read the API key from for
    this provider."""

    api_key_required: bool = True
    """Whether or not this provider's API requires an API key to be set.
    
    This can be False in cases such as local model providers (Ollama)."""

    api_key_default: str | None = None
    """A default API key to use for this provider if none is found in the
    environment variable specified by `api_key_env`."""

    supported_adapters: frozenset[str] = frozenset()
    """A set of adapter class names that this provider can be used along with
    to make requests to its API.
    """

    supported_language_model_prefixes: frozenset[str] = frozenset()
    """A set of language model name prefixes that are supported by this
    provider. This is used during broad model name inference."""

    supported_embedding_model_prefixes: frozenset[str] = frozenset()
    """A set of embedding model name prefixes that are supported by this
    provider. This is used during broad model name inference."""

    @property
    def is_custom(self) -> bool:
        """Returns True if this provider is a 'custom' provider
        (i.e. has a name that starts with 'custom:')."""
        return self.name.startswith("custom:")

    def clean_model(self, model: str) -> str:
        """Returns a cleaned version of the given model name for a
        request to this provider's API."""

        if self.name == "unkown" or self.name.startswith("custom:"):
            return model

        if model.startswith(f"{self.name}/"):
            return model[len(self.name) + 1 :]
        return model

    def get_api_key(self, api_key: str | None = None) -> str | None:
        """Retrieves the API key for this model provider.

        The order of precedence is as follows:
        1. The provided `api_key` argument.
        2. The environment variable specified by `api_key_env`.
        3. The `default_api_key` if set.

        Args:
            api_key (str | None): An optional API key to use.

        Returns:
            str | None: The resolved API key, or `None` if not found.
        """
        if api_key is not None:
            return api_key

        env_api_key = os.getenv(
            self.api_key_env,
            self.api_key_default if self.api_key_default is not None else None,
        )
        if self.api_key_required is True and env_api_key is None:
            raise ModelProviderException(
                f"API key is required for provider '{self.name}', but none was provided "
                f"and the environment variable '{self.api_key_env}' is not set."
            )
        return env_api_key


class ModelProvider:
    """Helper utility class object that is used simply to provide and register
    AI model API provider information objects within the libraries main
    MODEL_PROVIDERS_REGISTRY.
    """

    @classmethod
    def __class_getitem__(
        cls,
        params: Tuple[str, str | None, str | None, Literal["language", "embedding"]]
        | str,
    ) -> ModelProviderInfo:
        """Allows direct subscript notation access to the `infer_model_provider`
        method.

        This can be used as follows:

        `ModelProvider[model_name, base_url=optional_base_url, api_key=optional_api_key]`
        """
        if isinstance(params, str):
            model_name = params
            base_url = None
            api_key = None
            kind = "language"
        else:
            model_name, base_url, api_key, kind = params

        if kind is None:
            kind = "language"

        if model_name is None:
            raise ModelProviderInferenceException(
                model=None,
                base_url=base_url,
                message="Cannot infer model provider when no model name is given. Please include a model name"
                "as the first parameter when using the ModelProvider[...] syntax.",
            )

        return cls.infer_model_provider(
            model=model_name, base_url=base_url, api_key=api_key, kind=kind
        )

    @classmethod
    @lru_cache(maxsize=128)
    def infer_model_provider(
        cls,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        kind: Literal["language", "embedding"] = "language",
    ) -> ModelProviderInfo:
        """Infers an appopriate model provider info object for a given
        model name and/or optional base URL. If a custom base URL provider
        is not found, this creates a new 'custom' provider info object
        for the given base URL.

        This method can also be used by directly accessing the
        `__class_getitem__` method on the `ModelProvider` class:

        `ModelProvider[model_name, base_url=optional_base_url, api_key=optional_api_key]`

        Otherwise, all unknown model types or names will return an
        `unknown` provider info object, which will route to LiteLLM
        directly.

        NOTE: Providers marked as 'unknown' are not added to the registry,
        unlike custom providers.
        """
        if base_url is not None:
            return ModelProvider.custom_provider(base_url=base_url, api_key=api_key)

        if "/" in model:
            provider_name, model_name = model.split("/", 1)

            if provider_name in ModelProvider.list():
                return ModelProvider.get_provider(provider_name)

        # check against known model prefixes if all else fails
        for provider in MODEL_PROVIDERS_REGISTRY.values():
            prefixes = (
                provider.supported_language_model_prefixes
                if kind == "language"
                else provider.supported_embedding_model_prefixes
            )
            for prefix in prefixes:
                if model.startswith(prefix):
                    return provider

        else:
            return ModelProviderInfo(
                name="unknown",
                base_url=None,
                api_key_env=None,
                api_key_default=None,
                api_key_required=False,
                supported_adapters=frozenset({"litellm"}),
            )

    @classmethod
    def list(cls) -> List[str | ModelProviderName]:
        """Lists all currently registered AI providers within the
        registry."""
        global MODEL_PROVIDERS_REGISTRY
        keys = list(MODEL_PROVIDERS_REGISTRY.keys())
        keys.sort(key=lambda x: (x.startswith("custom:"), x))
        return keys

    @classmethod
    def update_provider(cls, provider: str, info: ModelProviderInfo) -> None:
        """Updates an existing AI provider info object within the registry."""
        global MODEL_PROVIDERS_REGISTRY

        if not provider in MODEL_PROVIDERS_REGISTRY:
            raise ModelProviderException(
                f"Cannot update AI provider '{provider}' as it does not exist "
                "in the providers registry."
            )
        MODEL_PROVIDERS_REGISTRY[provider] = info

    @classmethod
    @lru_cache(maxsize=128)
    def get_provider(
        cls, provider: str | ModelProviderName
    ) -> ModelProviderName | None:
        """Retrieve a registered AI provider info object from the
        registry by it's name.

        This raises an exception if the provider is not found.
        """
        global MODEL_PROVIDERS_REGISTRY

        if not provider in ModelProvider.list():
            raise ModelProviderException(
                f"AI provider '{provider}' is not registered in the providers registry."
            )
        return MODEL_PROVIDERS_REGISTRY[provider]

    @classmethod
    @lru_cache(maxsize=128)
    def custom_provider(
        cls, base_url: str, api_key: str | None = None
    ) -> ModelProviderInfo:
        """Creates a 'ready-to-go' custom AI provider info object based
        on a given base URL and optional API key. This is used when
        calling models with custom base URLs.

        This keys the provider name as 'custom:[base_url]'.

        NOTE: Currently all custom providers must be OpenAI compatible.
        """
        global MODEL_PROVIDERS_REGISTRY

        provider_name = f"custom:{base_url}"

        if provider_name in MODEL_PROVIDERS_REGISTRY:
            return MODEL_PROVIDERS_REGISTRY[provider_name]

        provider_info = ModelProviderInfo(
            name=provider_name,
            base_url=base_url,
            api_key_env=None,
            api_key_required=api_key is None,
            api_key_default=api_key,
            supported_adapters=frozenset({"openai", "litellm"}),
        )
        MODEL_PROVIDERS_REGISTRY[provider_name] = provider_info

        return provider_info


def _register_providers():
    """Performs initialization or first time registration of all
    known model providers within the registry."""

    from .anthropic import ANTHROPIC_MODEL_PROVIDERS
    from .openai import OPENAI_MODEL_PROVIDERS

    global MODEL_PROVIDERS_REGISTRY
    MODEL_PROVIDERS_REGISTRY.update(ANTHROPIC_MODEL_PROVIDERS)
    MODEL_PROVIDERS_REGISTRY.update(OPENAI_MODEL_PROVIDERS)


# NOTE:
# registration takes place right here!
_register_providers()
