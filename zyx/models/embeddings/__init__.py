"""zyx.models.embeddings"""

from ..._internal import _import_utils
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import EmbeddingModel
    from .types import (
        EmbeddingModelName,
        EmbeddingEncodingFormat,
    )


__all__ = [
    "EmbeddingModel",
    "EmbeddingModelName",
    "EmbeddingEncodingFormat",
]


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
