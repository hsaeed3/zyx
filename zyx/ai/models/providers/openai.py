"""zyx.ai.providers.openai

Registers ProviderInfo definitions for all OpenAI-compatible
model API providers supported in `zyx.ai`."""

from __future__ import annotations

from typing import Dict, TypeAliasType, Literal

from ..providers import (
    ModelProviderInfo,
    ModelProviderName,
)


OpenAIModelProviderName = TypeAliasType(
    "OpenAIModelProviderName",
    Literal[
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
"""All supported (known) OpenAI-compatible
model API provider names."""


OPENAI_MODEL_PROVIDERS: Dict[str | ModelProviderName, ModelProviderInfo] = {
    "cerebras": ModelProviderInfo(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        api_key_required=True,
        api_key_default=None,
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
    "cohere": ModelProviderInfo(
        name="cohere",
        base_url="https://api.cohere.ai/compatibility/v1",
        api_key_env="COHERE_API_KEY",
        api_key_required=True,
        api_key_default=None,
        supported_language_model_prefixes=frozenset({"command-", "xlarge-", "base-"}),
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
    "deepseek": ModelProviderInfo(
        name="deepseek",
        base_url="https://api.deepseek.ai/v1",
        api_key_env="DEEPSEEK_API_KEY",
        api_key_required=True,
        api_key_default=None,
        supported_language_model_prefixes=frozenset({"deepseek-"}),
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
    "fireworks": ModelProviderInfo(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        api_key_required=True,
        api_key_default=None,
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
    "groq": ModelProviderInfo(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        api_key_required=True,
        api_key_default=None,
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
    "lm_studio": ModelProviderInfo(
        name="lm_studio",
        base_url="http://localhost:1234/v1",
        api_key_env="LM_STUDIO_API_KEY",
        api_key_required=False,
        api_key_default="lm-studio-default-key",
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
    "moonshotai": ModelProviderInfo(
        name="moonshotai",
        base_url="https://api.moonshotai.com/v1",
        api_key_env="MOONSHOTAI_API_KEY",
        api_key_required=True,
        api_key_default=None,
        supported_language_model_prefixes=frozenset({"kimi-"}),
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
    "openai": ModelProviderInfo(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        api_key_required=True,
        api_key_default=None,
        supported_language_model_prefixes=frozenset(
            {"gpt-", "o1", "o3", "o4", "codex-", "chatgpt-"}
        ),
        supported_embedding_model_prefixes=frozenset({"text-embedding-"}),
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
    "ollama": ModelProviderInfo(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        api_key_required=False,
        api_key_default="ollama-default-key",
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
    "xai": ModelProviderInfo(
        name="xai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        api_key_required=True,
        api_key_default=None,
        supported_language_model_prefixes=frozenset({"grok-"}),
        supported_adapters=frozenset({"openai", "litellm"}),
    ),
}
"""Mapping of all 'known' or preconfigured AI model API
providers that are compatible with the OpenAI API specification.
"""
