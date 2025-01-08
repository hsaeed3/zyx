"""
zyx.types.client.client_provider_recommendation

Contains the `ClientProviderRecommendation` model, which is used to recommend the
best client provider to use for a given model and set of core kwargs.
"""

from __future__ import annotations

# [Imports]
from pydantic import BaseModel, ConfigDict
from typing import Literal, Optional


# ===================================================================
# [Client Provider Recommendation Model]
# ===================================================================


class ClientProviderRecommendation(BaseModel):
    """
    A model for recommending the best client provider to use for a given model
    and set of core kwargs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Literal["openai", "litellm"]
    """The provider to use for the client."""

    model: Optional[str] = None
    """If the model name was altered during the recommendation"""

    base_url: Optional[str] = None
    """If the base URL was altered during the recommendation"""

    api_key: Optional[str] = None
    """If the API key was altered during the recommendation"""

    # ===================================================================
    # [Helper Method]
    # ===================================================================

    @staticmethod
    def recommend(
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> ClientProviderRecommendation:
        """
        Recommends the best client provider to use for a given model and set of core
        kwargs.
        """

        # [check for base URL & instantly return if found]
        if base_url is not None:
            return ClientProviderRecommendation(provider="openai", base_url=base_url, api_key=api_key)

        # [check for OpenAI provider or model prefix]
        if model.startswith("gpt-") or model.startswith("openai/") or model.startswith("o1-"):
            # clear openai/ tag
            model = model.replace("openai/", "")
            return ClientProviderRecommendation(provider="openai", model=model)

        # [check for ollama or lmstudio]
        if model.startswith("ollama/"):
            # clear ollama/ tag
            model = model.replace("ollama/", "")
            return ClientProviderRecommendation(
                provider="openai",
                model=model,
                base_url="http://localhost:11434/v1",
                api_key="ollama" if not api_key else api_key,
            )
        elif model.startswith("lm_studio/"):
            # clear lm_studio/ tag
            model = model.replace("lm_studio/", "")
            return ClientProviderRecommendation(
                provider="openai",
                model=model,
                base_url="http://localhost:1234/v1",
                api_key="lmstudio" if not api_key else api_key,
            )

        # [Else recommend LiteLLM]
        else:
            return ClientProviderRecommendation(provider="litellm", model=model)


# ===================================================================
# [Test]
# ===================================================================


if __name__ == "__main__":
    # OpenAI base test
    openai_model = "gpt-4o"

    recommendation = ClientProviderRecommendation.recommend(model=openai_model)
    print(recommendation)

    assert recommendation.provider == "openai"

    # Ollama test
    ollama_model = "ollama/llama3.1"

    recommendation = ClientProviderRecommendation.recommend(model=ollama_model)
    print(recommendation)

    assert recommendation.provider == "openai"
    assert recommendation.model == "llama3.1"
    assert recommendation.base_url == "http://localhost:11434/v1"
    assert recommendation.api_key == "ollama"

    # Anthropic test
    anthropic_model = "anthropic/claude-3-5-sonnet-20240620"

    recommendation = ClientProviderRecommendation.recommend(model=anthropic_model)
    print(recommendation)

    assert recommendation.provider == "litellm"
    assert recommendation.model == "anthropic/claude-3-5-sonnet-20240620"
