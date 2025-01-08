"""
zyx.base.types.chat_completions.chat_model

`ease of use` type providing a list of commonly used provider &
model name combination strings.
"""

from __future__ import annotations

# [Imports]
from typing import Literal


# [Chat Model]
ChatModel = Literal[
    # Anthropic
    "anthropic/claude-3-5-haiku-latest",  # Vision
    "anthropic/claude-3-5-sonnet-latest",  # Vision
    "anthropic/claude-3-opus-latest",
    # Cohere
    "cohere/command-r-plus",
    "cohere/command-r",
    # Databricks
    "databricks/databricks-dbrx-instruct",
    # Gemini
    "gemini/gemini-pro",
    "gemini/gemini-1.5-pro-latest",
    # OpenAI
    "openai/gpt-4o-mini",  # Vision
    "openai/gpt-4o",  # Vision
    "openai/o1-mini",
    "openai/o1-preview",
    "openai/chatgpt-4o-latest",
    "openai/gpt-4-turbo",
    "openai/gpt-4",
    "openai/gpt-4-vision",  # Vision
    "openai/gpt-3.5-turbo",
    # Ollama
    "ollama/bespoke-minicheck",
    "ollama/llama3",
    "ollama/llama3.1",
    "ollama/llama3.2",
    "ollama/llama3.2-vision",  # Vision
    "ollama/llama-guard3",
    "ollama/llava",  # Vision
    "ollama/llava-llama3",  # Vision
    "ollama/llava-phi3",  # Vision
    "ollama/gemma2",
    "ollama/granite3-dense",
    "ollama/granite3-guardian",
    "ollama/granite3-moe",
    "ollama/minicpm-v",  # Vision
    "ollama/mistral",
    "ollama/mistral-nemo",
    "ollama/mistral-small",
    "ollama/mixtral",
    "ollama/moondream",  # Vision
    "ollama/nemotron",
    "ollama/nuextract",
    "ollama/opencoder",
    "ollama/phi3",
    "ollama/reader-lm",
    "ollama/smollm2",
    "ollama/shieldgemma",
    "ollama/tinyllama",
    "ollama/qwen",
    "ollama/qwen2" "ollama/qwen2.5",
    # Perplexity
    "perplexity/pplx-7b-chat",
    "perplexity/pplx-70b-chat",
    "perplexity/pplx-7b-online",
    "perplexity/pplx-70b-online",
    # XAI
    "xai/grok-beta",
]
