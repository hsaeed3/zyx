"""zyx.core"""

from typing import TYPE_CHECKING

from .._lib import _import_utils

if TYPE_CHECKING:
    from .schemas.schema import schema


__all__ = [
    # zyx.core.schemas.schema
    "schema"
]


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
