"""zyx.core.memory"""

from typing import TYPE_CHECKING

from ..._lib import _import_utils

if TYPE_CHECKING:
    from .memory import Memory, arag, mem, rag
    from .types import (
        MemoryDistanceMetric,
        MemoryItem,
        MemoryQueryResponse,
        MemorySearchResult,
        MemorySettings,
    )

__all__ = [
    "Memory",
    "mem",
    "rag",
    "arag",
    "MemoryItem",
    "MemorySearchResult",
    "MemoryQueryResponse",
    "MemorySettings",
    "MemoryDistanceMetric",
]


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
