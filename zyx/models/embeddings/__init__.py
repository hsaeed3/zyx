"""zyx.models.embeddings"""

from ...core.utils._import_utils import *

if TYPE_CHECKING:
    from .model import EmbeddingModel
    from .types import (
        EmbeddingModelSettings,
        EmbeddingModelName,
        EmbeddingModelResponse,
    )


__all__ = [
    "EmbeddingModel",
    "EmbeddingModelSettings",
    "EmbeddingModelName",
    "EmbeddingModelResponse",
]


__getattr__ = type_checking_getattr_fn(__all__)
__dir__ = type_checking_dir_fn(__all__)
