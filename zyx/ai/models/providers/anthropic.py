"""zyx.ai.providers.anthropic"""

from __future__ import annotations

from typing import Dict, TypeAliasType, Literal

from ..providers import (
    ModelProviderInfo,
    ModelProviderName,
)


AnthropicModelProviderName = TypeAliasType(
    "AnthropicProviderName", Literal["anthropic",]
)
"""Alias of all supported (known) Anthropic-compatible
model API provider names.

NOTE:
Currently the 'anthropic' ai adapter is only planned for
development, all anthropic models will currently route
directly to LiteLLM."""


ANTHROPIC_MODEL_PROVIDERS: Dict[str | ModelProviderName, ModelProviderInfo] = {
    "anthropic": ModelProviderInfo(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        api_key_required=True,
        api_key_default=None,
        supported_language_model_prefixes=frozenset({"claude-"}),
        supported_adapters=frozenset({"anthropic", "litellm"}),
    ),
}
