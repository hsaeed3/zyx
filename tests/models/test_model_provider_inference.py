import pytest
from zyx.models import providers


# Test infer_language_model_provider
def test_infer_language_model_provider_with_prefix():
    provider = providers.infer_language_model_provider("openai/gpt-4")
    assert provider is not None
    assert provider.name == "openai"


def test_infer_language_model_provider_with_known_prefix():
    provider = providers.infer_language_model_provider("gpt-4")
    assert provider is not None
    assert provider.name == "openai"


def test_infer_language_model_provider_with_cohere_prefix():
    provider = providers.infer_language_model_provider("command-r")
    assert provider is not None
    assert provider.name == "cohere"


def test_infer_language_model_provider_unknown():
    provider = providers.infer_language_model_provider("unknown-model")
    assert provider is None


def test_infer_language_model_provider_custom_prefix():
    provider = providers.infer_language_model_provider("custom/model")
    assert provider is None  # Since "custom" not in MODEL_PROVIDERS as a key


# Test infer_embedding_model_provider
def test_infer_embedding_model_provider_with_prefix():
    provider = providers.infer_embedding_model_provider("openai/text-embedding-ada-002")
    assert provider is not None
    assert provider.name == "openai"


def test_infer_embedding_model_provider_with_known_prefix():
    provider = providers.infer_embedding_model_provider("text-embedding-ada-002")
    assert provider is not None
    assert provider.name == "openai"


def test_infer_embedding_model_provider_unknown():
    provider = providers.infer_embedding_model_provider("unknown-embedding")
    assert provider is None


# Test custom_model_provider
def test_custom_model_provider_new():
    base_url = "https://custom.api/v1"
    api_key = "test_key"
    provider = providers.custom_model_provider(base_url, api_key)
    assert provider.name == base_url
    assert provider.base_url == base_url
    assert provider.default_api_key == api_key
    assert provider.requires_api_key is True
    assert base_url in providers.MODEL_PROVIDERS


def test_custom_model_provider_existing():
    base_url = "https://custom.api/v1"
    provider1 = providers.custom_model_provider(base_url)
    provider2 = providers.custom_model_provider(base_url)
    assert provider1 is provider2


# Test ModelProvider methods
def test_model_provider_clean_model_with_prefix():
    provider = providers.MODEL_PROVIDERS["openai"]
    cleaned = provider.clean_model("openai/gpt-4")
    assert cleaned == "gpt-4"


def test_model_provider_clean_model_without_prefix():
    provider = providers.MODEL_PROVIDERS["openai"]
    cleaned = provider.clean_model("gpt-4")
    assert cleaned == "gpt-4"


def test_model_provider_clean_model_custom():
    provider = providers.ModelProvider(
        name="custom",
        base_url="https://custom.com",
        api_key_env="",
        requires_api_key=False,
    )
    cleaned = provider.clean_model("custom/model")
    assert cleaned == "custom/model"


def test_model_provider_get_api_key_provided():
    provider = providers.MODEL_PROVIDERS["openai"]
    key = provider.get_api_key("provided_key")
    assert key == "provided_key"


def test_model_provider_get_api_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")
    provider = providers.MODEL_PROVIDERS["openai"]
    key = provider.get_api_key()
    assert key == "env_key"


def test_model_provider_get_api_key_default():
    provider = providers.MODEL_PROVIDERS["ollama"]
    key = provider.get_api_key()
    assert key == "ollama"


def test_model_provider_get_api_key_missing_required():
    provider = providers.MODEL_PROVIDERS["openai"]
    import os

    os.environ.pop("OPENAI_API_KEY", None)
    with pytest.raises(ValueError, match="API key is required"):
        provider.get_api_key()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
