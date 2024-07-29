# zyx =========================================================================

__all__ = [
    "db",
    "DB",
    "Document",
]

from ..core import _UtilLazyLoader 

class db(_UtilLazyLoader):
    pass
db.init("zyx.data", "db")

class DB(_UtilLazyLoader):
    pass
DB.init("zyx.data", "DB")

class Document(_UtilLazyLoader):
    pass
Document.init("zyx.data", "_Document")