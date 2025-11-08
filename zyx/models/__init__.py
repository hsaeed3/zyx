"""zyx.models"""

from .._internal import _import_utils
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .embeddings.model import EmbeddingModel
    from .embeddings.types import EmbeddingModelName, EmbeddingEncodingFormat

    from .language.model import LanguageModel
    from .language.types import (
        LanguageModelName,
        LanguageModelSettings,
        LanguageModelResponse,
    )


__all__ = [
    # zyx.models.embeddings
    "EmbeddingModel",
    "EmbeddingModelName",
    "EmbeddingEncodingFormat",
    # zyx.models.language
    "LanguageModel",
    "LanguageModelName",
    "LanguageModelSettings",
    "LanguageModelResponse",
]


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
