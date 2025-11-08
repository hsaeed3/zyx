"""zyx.models.embeddings.types"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAliasType

from ..definition import ModelSettings

__all__ = [
    "EmbeddingModelSettings",
    "EmbeddingEncodingFormat",
    "EmbeddingModelName",
]


@dataclass
class EmbeddingModelSettings(ModelSettings):
    """A definition of an embedding model, and associated settings."""

    @property
    def kind(self) -> str:
        return "embedding_model"

    dimensions: int | None = None
    """The dimensions to generate embeddings for."""
    encoding_format: EmbeddingEncodingFormat | None = None
    """The format to return the embeddings in."""
    user: str | None = None
    """The user to generate embeddings for."""


EmbeddingEncodingFormat = TypeAliasType(
    "EmbeddingEncodingFormat",
    Literal["float", "base64"],
)
"""The format to return the embeddings in."""


EmbeddingModelName = TypeAliasType(
    "EmbeddingModelName",
    Literal[
        "mock",
        "openai/text-embedding-3-small",
        "openai/text-embedding-3-large",
        "openai/text-embedding-ada-002",
        # local
        "ollama/embeddinggemma",
        "ollama/nomic-embed-text",
        "ollama/mxbai-embed-large",
        "ollama/bge-m3",
        "ollama/all-minilm",
        "ollama/snowflake-arctic-embed",
        "ollama/snowflake-arctic-embed2",
        "ollama/bge-large",
        "ollama/paraphrase-multilingual",
        "ollama/granite-embedding",
        "ollama/qwen3-embedding",
    ],
)
"""Alias for common embedding model names. Unlike the LanguageModelName
alias, this contains only OpenAI's embedding models, and a collection of the
most popular local embedding models from Ollama. 
When using embedding models, (if your hardware permits it), you should prefer to use
them locally!
"""
