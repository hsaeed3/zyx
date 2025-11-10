"""zyx.core.stores

Store implementations for data persistence with optional embeddings.
"""

from typing import TYPE_CHECKING

from ..._internal import _import_utils

if TYPE_CHECKING:
    from .store import Store
    from .type import TypeStore
    from .types import (
        DEFAULT_STORE_TABLE_NAME,
        StoreDistanceMetric,
        StoreItem,
        StoreLocationName,
        StoreSearchResult,
        StoreSettings,
        TypeEntry,
    )
    from .utils import generate_content_key, json_ready, stable_string

__all__ = [
    "Store",
    "TypeStore",
    "DEFAULT_STORE_TABLE_NAME",
    "StoreLocationName",
    "StoreDistanceMetric",
    "StoreSettings",
    "StoreItem",
    "StoreSearchResult",
    "TypeEntry",
    "generate_content_key",
    "stable_string",
    "json_ready",
]


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
