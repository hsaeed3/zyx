"""zyx.core.models"""

from typing import TYPE_CHECKING

from ..._internal import _import_utils

if TYPE_CHECKING:
    from .clients import ModelClient
    from .embeddings.model import (  # NOTE: this is a direct openai re-export
        ChonkieEmbeddingModel,
        EmbeddingModel,
        EmbeddingModelResponse,
    )
    from .embeddings.types import (
        EmbeddingModelName,
        EmbeddingModelSettings,
    )
    from .language.model import LanguageModel
    from .language.types import (
        LanguageModelName,
        LanguageModelResponse,
        LanguageModelSettings,
    )
    from .providers import (
        ModelProvider,
        ModelProviderName,
        ModelProviderRegistry,
    )


__all__ = (
    "LanguageModel",
    "LanguageModelName",
    "LanguageModelSettings",
    "LanguageModelResponse",
    "EmbeddingModel",
    "ChonkieEmbeddingModel",
    "EmbeddingModelResponse",
    "EmbeddingModelName",
    "EmbeddingModelSettings",
    "ModelProvider",
    "ModelProviderName",
    "ModelProviderRegistry",
    "ModelClient",
)


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
