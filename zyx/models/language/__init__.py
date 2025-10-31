"""zyx.models.language"""

from ...core.utils._import_utils import *

if TYPE_CHECKING:
    from .model import LanguageModel
    from .types import LanguageModelSettings, LanguageModelName, LanguageModelResponse


__all__ = [
    "LanguageModel",
    "LanguageModelSettings",
    "LanguageModelName",
    "LanguageModelResponse",
]


__getattr__ = type_checking_getattr_fn(__all__)
__dir__ = type_checking_dir_fn(__all__)
