"""zyx.core.models.providers"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, Literal, TypeAliasType, TypedDict

ModelProviderName = TypeAliasType(
    "ModelProviderName",
    Literal[
        "cerebras",
        "cohere",
        "deepseek",
        "fireworks",
        "groq",
        "lm_studio",
        "moonshotai",
        "ollama",
        "openai",
        "openrouter",
        "xai",
    ],
)
"""All explicitly named model providers supported within the ZYX framework.

*Most other model providers are supported by the LiteLLM client, if installed.*
"""


class ModelProviderSupportedPrefixes(TypedDict):
    """Dictionary representation of supported model prefixes that can be used to
    identify a model provider through an arbitrary model name."""

    language_model: FrozenSet[str] = field(default_factory=frozenset)
    """A set of prefixes that can be used to identify a language model."""

    embedding_model: FrozenSet[str] = field(default_factory=frozenset)
    """A set of prefixes that can be used to identify an embedding model."""


@dataclass(frozen=True)
class ModelProvider:
    """Definition class for a known or custom configured model provider. A model
    provider is the representation of metadata and configuration parameters to use
    for a specific API provider (OpenAI, OpenRouter, etc.)
    """

    name: str
    """The name of the model provider."""

    base_url: str | None = None
    """The base URL of the model provider."""

    api_key_env_var: str | None = None
    """An environment variable that *might* contain the API key for the model provider."""

    api_key_required: bool = True
    """Whether or not the API key is required to make requests to the model provider."""

    api_key_default: str | None = None
    """A default API key to use if no API key is provided."""

    supported_clients: FrozenSet[str] = field(default_factory=frozenset)
    """A list of model client class names that this supporter can be routed through.

    NOTE: LiteLLM is never included in this set for any provider, as we directly assume
    it is supported by default.
    """

    supported_prefixes: ModelProviderSupportedPrefixes = field(
        default_factory=ModelProviderSupportedPrefixes
    )
    """A dictionary of supported model prefixes for this provider."""

    @property
    def is_custom(self) -> bool:
        """Check if this is a custom provider (not preconfigured)."""
        return self.name.startswith("custom:")

    @property
    def is_local(self) -> bool:
        """Check if this provider runs locally (no API key required)."""
        return (
            not self.api_key_required and self.api_key_default is not None
        )

    def clean_model_name(self, model: str) -> str:
        """Remove provider prefix from model name if present.

        Args:
            model: Model name potentially prefixed with provider

        Returns:
            Clean model name without provider prefix

        Examples:
            >>> provider.clean_model_name("openai/gpt-4")
            "gpt-4"
            >>> provider.clean_model_name("gpt-4")
            "gpt-4"
        """
        if self.is_custom or self.name == "unknown":
            return model

        prefix = f"{self.name}/"
        return model[len(prefix) :] if model.startswith(prefix) else model

    def get_api_key(self) -> str | None:
        """Retrieve the API key for this provider.

        Resolution order:
        1. Environment variable (if api_key_env is set)
        2. Default API key (if api_key_default is set)
        3. None

        Returns:
            The resolved API key or None
        """
        if self.api_key_env_var:
            return os.getenv(self.api_key_env_var, self.api_key_default)
        return self.api_key_default


class ModelProviderRegistry:
    """Registry of all known or custom configured model providers, this
    is used to lookup and retrieve model providers by name or other metadata."""

    _INSTANCE: ModelProviderRegistry | None = None

    _PROVIDERS: Dict[str, ModelProvider] = field(default_factory=dict)
    """A dictionary of model provider names to their corresponding model provider objects."""

    def __new__(cls) -> ModelProviderRegistry:
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls)
        return cls._INSTANCE

    def __init__(self) -> None:
        # TODO:
        # this could go in a better spot? currently i have no real reason to
        # oop / separate this further, but in the case that a client such as
        # anthropic / any non-openai schema provider is explicitly added, this
        # may need to be split up a teeny bit.

        self._PROVIDERS = {
            "cerebras": ModelProvider(
                name="cerebras",
                base_url="https://api.cerebras.ai/v1",
                api_key_env_var="CEREBRAS_API_KEY",
                api_key_required=True,
                supported_clients=frozenset({"openai"}),
            ),
            "cohere": ModelProvider(
                name="cohere",
                base_url="https://api.cohere.ai/compatibility/v1",
                api_key_env_var="COHERE_API_KEY",
                api_key_required=True,
                supported_clients=frozenset({"openai"}),
                supported_prefixes=ModelProviderSupportedPrefixes(
                    language_model=frozenset(
                        {"command-", "xlarge-", "base-"}
                    )
                ),
            ),
            "deepseek": ModelProvider(
                name="deepseek",
                base_url="https://api.deepseek.ai/v1",
                api_key_env_var="DEEPSEEK_API_KEY",
                api_key_required=True,
                supported_clients=frozenset({"openai"}),
                supported_prefixes=ModelProviderSupportedPrefixes(
                    language_model=frozenset(
                        {
                            "deepseek-",
                        }
                    )
                ),
            ),
            "fireworks": ModelProvider(
                name="fireworks",
                base_url="https://api.fireworks.ai/inference/v1",
                api_key_env_var="FIREWORKS_API_KEY",
                api_key_required=True,
                supported_clients=frozenset({"openai"}),
            ),
            "groq": ModelProvider(
                name="groq",
                base_url="https://api.groq.com/openai/v1",
                api_key_env_var="GROQ_API_KEY",
                api_key_required=True,
                supported_clients=frozenset({"openai"}),
            ),
            "lm_studio": ModelProvider(
                name="lm_studio",
                base_url="http://localhost:1234/v1",
                api_key_env_var="LM_STUDIO_API_KEY",
                api_key_required=False,
                api_key_default="lm-studio-default-key",
                supported_clients=frozenset({"openai"}),
            ),
            "moonshotai": ModelProvider(
                name="moonshotai",
                base_url="https://api.moonshotai.com/v1",
                api_key_env_var="MOONSHOTAI_API_KEY",
                api_key_required=True,
                supported_clients=frozenset({"openai"}),
                supported_prefixes=ModelProviderSupportedPrefixes(
                    language_model=frozenset(
                        {
                            "kimi-",
                        }
                    )
                ),
            ),
            "ollama": ModelProvider(
                name="ollama",
                base_url="http://localhost:11434",
                api_key_env_var="OLLAMA_API_KEY",
                api_key_required=False,
                api_key_default="ollama-default-key",
                supported_clients=frozenset({"openai"}),
            ),
            "openai": ModelProvider(
                name="openai",
                base_url="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                api_key_required=True,
                supported_clients=frozenset({"openai"}),
                supported_prefixes=ModelProviderSupportedPrefixes(
                    language_model=frozenset(
                        {"gpt-", "o1", "o3", "o4", "codex-", "chatgpt-"}
                    )
                ),
            ),
            "openrouter": ModelProvider(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key_env_var="OPENROUTER_API_KEY",
                api_key_required=True,
                supported_clients=frozenset({"openai"}),
            ),
            "xai": ModelProvider(
                name="xai",
                base_url="https://api.x.ai/v1",
                api_key_env_var="XAI_API_KEY",
                api_key_required=True,
                supported_clients=frozenset({"openai"}),
                supported_prefixes=ModelProviderSupportedPrefixes(
                    language_model=frozenset(
                        {
                            "grok-",
                        }
                    )
                ),
            ),
        }

    def get(self, name: ModelProviderName) -> ModelProvider | None:
        """Retrieve a model provider by name."""

        return self._PROVIDERS.get(name)

    def register_custom(
        self, base_url: str, api_key: str | None = None
    ) -> ModelProvider:
        self._PROVIDERS[f"custom:{base_url}"] = ModelProvider(
            name=f"custom:{base_url}",
            base_url=base_url,
            api_key_env_var=api_key,
            api_key_required=api_key is not None,
            api_key_default=api_key,
            supported_clients=frozenset({"openai"}),
        )

    @lru_cache(maxsize=1024)
    def infer_from_model_name(
        self,
        model: str,
        kind: Literal[
            "language_model", "embedding_model"
        ] = "language_model",
    ) -> ModelProvider | None:
        """Retrieve a model provider by inferring from a given name."""
        if "/" in model:
            provider_name, model_name = model.split("/", 1)

            if provider_name in self._PROVIDERS:
                return self._PROVIDERS[provider_name]

            # else: fallback

        for provider in self._PROVIDERS.values():
            for prefix in provider.supported_prefixes.get(
                kind, frozenset()
            ):
                if model.startswith(prefix):
                    return provider
        return None
