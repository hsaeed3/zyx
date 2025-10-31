"""zyx.models

Central location for interacting with generative AI models, including
generating chat completions and structured outputs through the `instructor`
library and various model client backends.
"""

from ..core.utils._import_utils import *

if TYPE_CHECKING:
    from .language.model import LanguageModel
    from .language.types import (
        LanguageModelName,
        LanguageModelSettings,
        LanguageModelResponse,
    )

    from .embeddings.model import EmbeddingModel
    from .embeddings.types import (
        EmbeddingModelName,
        EmbeddingModelSettings,
        EmbeddingModelResponse,
    )


__all__ = [
    "LanguageModel",
    "LanguageModelName",
    "LanguageModelSettings",
    "LanguageModelResponse",
    "EmbeddingModel",
    "EmbeddingModelName",
    "EmbeddingModelSettings",
    "EmbeddingModelResponse",
]


__getattr__ = type_checking_getattr_fn(__all__)
__dir__ = type_checking_dir_fn(__all__)
