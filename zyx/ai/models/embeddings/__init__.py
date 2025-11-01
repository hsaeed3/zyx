"""zyx.ai.models.embeddings"""

from ....core.utils._import_utils import *

if TYPE_CHECKING:
    from .model import EmbeddingModel, embed, aembed
    from .types import (
        EmbeddingModelName,
        EmbeddingModelResponse,
        EmbeddingModelSettings,
    )


__all__ = [
    "EmbeddingModel",
    "embed",
    "aembed",
    "EmbeddingModelName",
    "EmbeddingModelResponse",
    "EmbeddingModelSettings",
]


__getattr__ = type_checking_getattr_fn(__all__)
__dir__ = type_checking_dir_fn(__all__)
