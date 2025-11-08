"""zyx.models.embeddings"""

from typing import TYPE_CHECKING

from ..._lib import _import_utils

if TYPE_CHECKING:
    from .model import EmbeddingModel, arun_embed, embedder, run_embed
    from .types import EmbeddingEncodingFormat, EmbeddingModelName


__all__ = [
    "arun_embed",
    "run_embed",
    "embedder",
    "EmbeddingModel",
    "EmbeddingModelName",
    "EmbeddingEncodingFormat",
]


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
