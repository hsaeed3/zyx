import pytest
import os
from zyx.core.models import providers


# Test infer_from_model_name for language models
def test_infer_language_model_provider_with_prefix():
    """Test inference with explicit provider prefix (openai/gpt-4)"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name("openai/gpt-4")
    assert provider is not None
    assert provider.name == "openai"


def test_infer_language_model_provider_with_known_prefix():
    """Test inference with known model prefix (gpt-4)"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name("gpt-4")
    assert provider is not None
    assert provider.name == "openai"


def test_infer_language_model_provider_with_cohere_prefix():
    """Test inference with Cohere model prefix (command-r)"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name("command-r")
    assert provider is not None
    assert provider.name == "cohere"


def test_infer_language_model_provider_with_deepseek_prefix():
    """Test inference with DeepSeek model prefix"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name("deepseek-coder")
    assert provider is not None
    assert provider.name == "deepseek"


def test_infer_language_model_provider_with_grok_prefix():
    """Test inference with xAI Grok model prefix"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name("grok-1")
    assert provider is not None
    assert provider.name == "xai"


def test_infer_language_model_provider_with_kimi_prefix():
    """Test inference with MoonshotAI Kimi model prefix"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name("kimi-chat")
    assert provider is not None
    assert provider.name == "moonshotai"


def test_infer_language_model_provider_unknown():
    """Test that unknown model names return None"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name("unknown-model")
    assert provider is None


def test_infer_language_model_provider_custom_prefix():
    """Test that custom/unknown prefix returns None"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name("custom/model")
    assert provider is None


# Test infer_from_model_name for embedding models
def test_infer_embedding_model_provider_with_prefix():
    """Test embedding model inference with explicit provider prefix"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name(
        "openai/text-embedding-ada-002", kind="embedding_model"
    )
    assert provider is not None
    assert provider.name == "openai"


def test_infer_embedding_model_provider_unknown():
    """Test that unknown embedding model returns None"""
    registry = providers.ModelProviderRegistry()
    provider = registry.infer_from_model_name(
        "unknown-embedding", kind="embedding_model"
    )
    assert provider is None


# Test custom provider registration
def test_custom_model_provider_new():
    """Test registering a new custom provider"""
    registry = providers.ModelProviderRegistry()
    base_url = "https://custom.api/v1"
    api_key = "test_key"
    registry.register_custom(base_url, api_key)
    provider = registry.get(f"custom:{base_url}")
    assert provider is not None
    assert provider.name == f"custom:{base_url}"
    assert provider.base_url == base_url
    assert provider.api_key_default == api_key
    assert provider.api_key_required is True
    assert provider.is_custom


def test_custom_model_provider_without_api_key():
    """Test registering a custom provider without API key"""
    registry = providers.ModelProviderRegistry()
    base_url = "https://local.api/v1"
    registry.register_custom(base_url)
    provider = registry.get(f"custom:{base_url}")
    assert provider is not None
    assert provider.api_key_default is None
    assert provider.api_key_required is False


# Test ModelProvider methods
def test_model_provider_clean_model_with_prefix():
    """Test cleaning model name with provider prefix"""
    registry = providers.ModelProviderRegistry()
    provider = registry.get("openai")
    assert provider is not None
    cleaned = provider.clean_model_name("openai/gpt-4")
    assert cleaned == "gpt-4"


def test_model_provider_clean_model_without_prefix():
    """Test cleaning model name without provider prefix"""
    registry = providers.ModelProviderRegistry()
    provider = registry.get("openai")
    assert provider is not None
    cleaned = provider.clean_model_name("gpt-4")
    assert cleaned == "gpt-4"


def test_model_provider_clean_model_custom():
    """Test that custom provider doesn't clean model names"""
    registry = providers.ModelProviderRegistry()
    base_url = "https://custom.com"
    registry.register_custom(base_url)
    provider = registry.get(f"custom:{base_url}")
    assert provider is not None
    cleaned = provider.clean_model_name("custom/model")
    assert cleaned == "custom/model"


def test_model_provider_get_api_key_env(monkeypatch):
    """Test getting API key from environment variable"""
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")
    registry = providers.ModelProviderRegistry()
    provider = registry.get("openai")
    assert provider is not None
    key = provider.get_api_key()
    assert key == "env_key"


def test_model_provider_get_api_key_default():
    """Test getting default API key for local providers"""
    registry = providers.ModelProviderRegistry()
    provider = registry.get("ollama")
    assert provider is not None
    key = provider.get_api_key()
    assert key == "ollama-default-key"


def test_model_provider_get_api_key_missing():
    """Test getting API key when not set returns None"""
    registry = providers.ModelProviderRegistry()
    provider = registry.get("openai")
    assert provider is not None
    # Clear the environment variable
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        key = provider.get_api_key()
        assert key is None
    finally:
        # Restore if it existed
        if old_key is not None:
            os.environ["OPENAI_API_KEY"] = old_key


# Test provider properties
def test_model_provider_is_local():
    """Test is_local property for local providers"""
    registry = providers.ModelProviderRegistry()
    ollama = registry.get("ollama")
    openai = registry.get("openai")
    assert ollama is not None
    assert openai is not None
    assert ollama.is_local is True
    assert openai.is_local is False


def test_model_provider_is_custom():
    """Test is_custom property"""
    registry = providers.ModelProviderRegistry()
    registry.register_custom("https://custom.api")
    custom = registry.get("custom:https://custom.api")
    openai = registry.get("openai")
    assert custom is not None
    assert openai is not None
    assert custom.is_custom is True
    assert openai.is_custom is False


# Test registry.get method
def test_registry_get_existing_provider():
    """Test getting an existing provider from registry"""
    registry = providers.ModelProviderRegistry()
    provider = registry.get("openai")
    assert provider is not None
    assert provider.name == "openai"


def test_registry_get_nonexistent_provider():
    """Test getting a non-existent provider returns None"""
    registry = providers.ModelProviderRegistry()
    provider = registry.get("nonexistent")
    assert provider is None


# Test singleton pattern
def test_registry_singleton():
    """Test that ModelProviderRegistry is a singleton"""
    registry1 = providers.ModelProviderRegistry()
    registry2 = providers.ModelProviderRegistry()
    assert registry1 is registry2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
