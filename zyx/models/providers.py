"""zyx.models.providers"""

from enum import Enum
from functools import lru_cache
import os
from typing import Dict, Literal, TypeAliasType
from dataclasses import dataclass, field

from openai import AsyncOpenAI

__all__ = [
    "ModelProviderName",
    "ModelProviderType",
    "ModelProvider",
    "MODEL_PROVIDERS",
    "infer_language_model_provider",
    "infer_embedding_model_provider",
    "custom_model_provider",
]


ModelProviderName = TypeAliasType(
    "ModelProviderName",
    Literal[
        "openai",
        "cohere",
        "cerebras",
        "deepseek",
        "fireworks",
        "github",
        "groq",
        "lm_studio",
        "moonshotai",
        "ollama",
        "openrouter",
        "xai",
        # kept simply to define a provider type for 'custom'
        # no need to set 'custom' when setting a model provider
        # for a backend
    ],
)
"""Alias type representation of compatible provider names within `zyx`
that implement OpenAI API endpoints."""


class ModelProviderType(Enum):
    """An enumeration that represents all the 'known' or supported model
    providers within `zyx`.

    Models that are defined using a custom, user given base URL will be
    set as `CUSTOM`.
    All other models will fallback to `UNKNOWN`, these models will utilize the
    LiteLLM model client if available.
    """

    OPENAI = "openai"
    COHERE = "cohere"
    CEREBRAS = "cerebras"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    GITHUB = "github"
    GROQ = "groq"
    LM_STUDIO = "lm_studio"
    MOONSHOTAI = "moonshotai"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    XAI = "xai"

    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelProvider:
    """Simple metadata / configuration dataclass that represents a
    single ai model API provider.

    NOTE: A `provider` in the context of `zyx` represents any model
    API that is compatible with the OpenAI API specification. All other
    models fallback to use `LiteLLM` as it's model client, if the library
    is available.
    """

    name: str
    """The name of this model provider."""

    base_url: str
    """The base URL for this model provider's API."""

    api_key_env: str
    """The environment variable name where the API key for this model
    provider is stored."""

    requires_api_key: bool = True
    """Whether or not this model provider requires an API key. This can
    be `False` in the case of local model providers such as Ollama."""

    default_api_key: str | None = None
    """The default API key to use for this model provider if no API key is provided."""

    supported_language_model_prefixes: frozenset[str] = frozenset()
    """A frozenset of language model name prefixes that this provider supports.
    This is used only during model resolution, if no provider is explicitly
    given."""

    supported_embedding_model_prefixes: frozenset[str] = frozenset()
    """A frozenset of embedding model name prefixes that this provider
    supports. This is used only during model resolution, if no provider is
    explicitly given."""

    def clean_model(self, model: str) -> str:
        """
        Helper method that removes any provider prefix from a
        given model name if applicable.
        """
        if self.name == "custom" or self.name == "unknown":
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
            self.default_api_key if self.default_api_key is not None else None,
        )
        if self.requires_api_key and env_api_key is None:
            raise ValueError(
                f"API key is required for provider '{self.name}', but none was provided "
                f"and the environment variable '{self.api_key_env}' is not set."
            )
        return env_api_key


MODEL_PROVIDERS: Dict[ModelProviderName, ModelProvider] = {
    "cerebras": ModelProvider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset(),
        supported_embedding_model_prefixes=frozenset(),
    ),
    "cohere": ModelProvider(
        name="cohere",
        base_url="https://api.cohere.ai/compatibility/v1",
        api_key_env="COHERE_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset({"command-"}),
        # although cohere uses a shared 'embed-' prefix namespace
        # for embedding models, its just too generic tbh...
        supported_embedding_model_prefixes=frozenset(),
    ),
    "deepseek": ModelProvider(
        name="deepseek",
        base_url="https://api.deepseek.ai/v1",
        api_key_env="DEEPSEEK_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset({"deepseek-"}),
        supported_embedding_model_prefixes=frozenset(),
    ),
    "fireworks": ModelProvider(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset(),
        supported_embedding_model_prefixes=frozenset(),
    ),
    "github": ModelProvider(
        name="github",
        base_url="https://models.github.ai/inference",
        api_key_env="GITHUB_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset(),
        supported_embedding_model_prefixes=frozenset(),
    ),
    "groq": ModelProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset(),
        supported_embedding_model_prefixes=frozenset(),
    ),
    "lm_studio": ModelProvider(
        name="lm_studio",
        base_url="http://localhost:1234/v1",
        api_key_env="LM_STUDIO_API_KEY",
        requires_api_key=False,
        default_api_key="lmstudio",
        supported_language_model_prefixes=frozenset(),
        supported_embedding_model_prefixes=frozenset(),
    ),
    "moonshotai": ModelProvider(
        name="moonshotai",
        base_url="https://api.moonshot.ai/v1",
        api_key_env="MOONSHOT_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset({"kimi-"}),
        supported_embedding_model_prefixes=frozenset(),
    ),
    "ollama": ModelProvider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        requires_api_key=False,
        default_api_key="ollama",
        supported_language_model_prefixes=frozenset(),
        supported_embedding_model_prefixes=frozenset(),
    ),
    "openai": ModelProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset(
            {"gpt-", "o1", "o2", "o3", "o4", "codex-", "chatgpt-"}
        ),
        supported_embedding_model_prefixes=frozenset({"text-embedding-"}),
    ),
    "openrouter": ModelProvider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset(),
        supported_embedding_model_prefixes=frozenset(),
    ),
    "xai": ModelProvider(
        name="xai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        requires_api_key=True,
        supported_language_model_prefixes=frozenset({"grok-"}),
        supported_embedding_model_prefixes=frozenset(),
    ),
    # NOTE:
    # this is used just as a placeholder provider for
    # intantiatating a LiteLLM backend given no
    # provider information or api key (using litellm's
    # model inference instead)
    "LITELLM_BACKEND_DEFAULT": ModelProvider(
        name="LITELLM_BACKEND_DEFAULT",
        base_url="",
        api_key_env="",
        requires_api_key=False,
        supported_language_model_prefixes=frozenset(),
        supported_embedding_model_prefixes=frozenset(),
    ),
}
"""Dictionary mapping that provides metadata / configuration
for a specific OpenAI API compatible model provider."""


@lru_cache()
def infer_language_model_provider(model: str) -> ModelProvider | None:
    """Infers a model provider from the given model name.

    This provider only returns known model providers, if
    a model is not recognized, `None` is returned.

    Args:
        model (str): The model name to infer the provider from.

    Returns:
        ModelProvider | None: The inferred model provider, or
        `None` if no provider could be inferred.
    """

    if "/" in model:
        provider, model_name = model.split("/", 1)

        if provider in MODEL_PROVIDERS:
            return MODEL_PROVIDERS[provider]

    # non-prefixed
    for provider in MODEL_PROVIDERS.values():
        for prefix in provider.supported_language_model_prefixes:
            if model.startswith(prefix):
                return provider

    return None


@lru_cache()
def infer_embedding_model_provider(model: str) -> ModelProvider | None:
    """Infers a model provider from the given embedding model name.

    This provider only returns known model providers, if
    a model is not recognized, `None` is returned.

    Args:
        model (str): The embedding model name to infer the provider from.

    Returns:
        ModelProvider | None: The inferred model provider, or
        `None` if no provider could be inferred.
    """

    if "/" in model:
        provider, model_name = model.split("/", 1)

        if provider in MODEL_PROVIDERS:
            return MODEL_PROVIDERS[provider]

    # non-prefixed
    for provider in MODEL_PROVIDERS.values():
        for prefix in provider.supported_embedding_model_prefixes:
            if model.startswith(prefix):
                return provider

    return None


def custom_model_provider(base_url: str, api_key: str | None = None) -> ModelProvider:
    """Adds a custom model provider to the set of known model providers.

    This is used to define custom OpenAI API compatible model endpoints
    as a named provider for client caching and other operations.

    This keys the provider by the custom given `base_url`.
    """
    global MODEL_PROVIDERS

    if not MODEL_PROVIDERS.get(base_url, None):
        MODEL_PROVIDERS[base_url] = ModelProvider(
            name=base_url,
            base_url=base_url,
            api_key_env="",
            requires_api_key=api_key is not None,
            default_api_key=api_key,
            supported_language_model_prefixes=frozenset(),
            supported_embedding_model_prefixes=frozenset(),
        )
    return MODEL_PROVIDERS[base_url]
