"""zyx.models.embeddings.types"""

from typing import TypeAliasType, Literal

__all__ = [
    "EmbeddingEncodingFormat",
    "EmbeddingModelName",
]


EmbeddingEncodingFormat = TypeAliasType(
    "EmbeddingEncodingFormat",
    Literal["float", "base64"],
)
"""The format to return the embeddings in."""


EmbeddingModelName = TypeAliasType(
    "EmbeddingModelName",
    Literal[
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
