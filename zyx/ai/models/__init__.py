"""zyx.ai.models

Primary namespace for `Model` interfaces. A `Model` is a simple interface
for interacting with a specific kind of AI model. All `Model` implementations
provide a `.run()` and `.arun()` method for synchronous and asynchronous
invocation respectively, with streaming support where applicable.
"""

from ...core.utils._import_utils import *

if TYPE_CHECKING:
    from .language.model import LanguageModel, llm, allm
    from .language.types import (
        LanguageModelName,
        LanguageModelSettings,
        LanguageModelResponse,
    )

    from .embeddings.model import EmbeddingModel, embed, aembed
    from .embeddings.types import (
        EmbeddingModelName,
        EmbeddingModelSettings,
        EmbeddingModelResponse,
    )

    from .providers import ModelProvider, ModelProviderInfo
    from .adapters import ModelAdapter


__all__ = [
    "LanguageModel",
    "llm",
    "allm",
    "LanguageModelName",
    "LanguageModelSettings",
    "LanguageModelResponse",
    "EmbeddingModel",
    "embed",
    "aembed",
    "EmbeddingModelName",
    "EmbeddingModelSettings",
    "EmbeddingModelResponse",
    "ModelProvider",
    "ModelProviderInfo",
    "ModelAdapter",
]


__getattr__ = type_checking_getattr_fn(__all__)
__dir__ = type_checking_dir_fn(__all__)
