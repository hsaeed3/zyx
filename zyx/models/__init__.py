"""zyx.models"""

from typing import TYPE_CHECKING

from .._lib import _import_utils

if TYPE_CHECKING:
    from .clients import ModelClient
    from .definition import ModelDefinition
    from .embeddings.model import EmbeddingModel, arun_embed, embedder, run_embed
    from .embeddings.types import (
        EmbeddingEncodingFormat,
        EmbeddingModelName,
        EmbeddingModelSettings,
    )
    from .language.model import LanguageModel, arun_llm, llm, run_llm
    from .language.types import (
        LanguageModelName,
        LanguageModelResponse,
        LanguageModelSettings,
    )
    from .providers import ModelProvider, ModelProviderRegistry


__all__ = [
    # zyx.models.definition
    "ModelDefinition",
    # zyx.models.providers
    "ModelProvider",
    "ModelProviderRegistry",
    # zyx.models.clients
    "ModelClient",
    # zyx.models.language
    "LanguageModel",
    "llm",
    "run_llm",
    "arun_llm",
    "LanguageModelName",
    "LanguageModelResponse",
    "LanguageModelSettings",
    # zyx.models.embeddings
    "EmbeddingModel",
    "embedder",
    "run_embed",
    "arun_embed",
    "EmbeddingModelName",
    "EmbeddingEncodingFormat",
    "EmbeddingModelSettings",
]


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
