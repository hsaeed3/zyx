"""zyx.ai.utils"""

from ...core.utils._import_utils import *

if TYPE_CHECKING:
    from .function_calling import openai_function_schema
    from .structured_outputs import prepare_structured_output_model


__all__ = [
    "openai_function_schema",
    "prepare_structured_output_model",
]


__getattr__ = type_checking_getattr_fn(__name__)
__dir__ = type_checking_dir_fn(__name__)
