"""⚡︎ zyx"""

from .core.utils._import_utils import *

if TYPE_CHECKING:
    # direct model runnables & utils
    from .ai.models.language.model import llm, allm
    from .ai.models.embeddings.model import embed, aembed
    from .ai.utils.function_calling import openai_function_schema
    from .ai.utils.structured_outputs import prepare_structured_output_model


__all__ = [
    # CORE METHODS
    "llm",
    "allm",
    "embed",
    "aembed",
    # UTILS
    "openai_function_schema",
    "prepare_structured_output_model",
]


__getattr__ = type_checking_getattr_fn(__all__)
__dir__ = type_checking_dir_fn(__all__)
