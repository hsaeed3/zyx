"""zyx.ai.models.language"""

from ....core.utils._import_utils import *

if TYPE_CHECKING:
    from .model import LanguageModel, llm, allm
    from .types import (
        LanguageModelName,
        LanguageModelResponse,
        LanguageModelSettings,
    )


__all__ = [
    "LanguageModel",
    "llm",
    "allm",
    "LanguageModelName",
    "LanguageModelResponse",
    "LanguageModelSettings",
]


__getattr__ = type_checking_getattr_fn(__all__)
__dir__ = type_checking_dir_fn(__all__)
