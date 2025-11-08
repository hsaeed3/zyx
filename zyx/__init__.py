"""⚡️ zyx"""

from typing import TYPE_CHECKING

from ._lib import _import_utils

# NOTE:
# this is a 'maybe?' design choice for now,
# all top level exports are **all** functions,
# class level imports will be module to module.

if TYPE_CHECKING:
    # --- core
    from .core.schemas.schema import to_schema
    from .core.memory.memory import mem, rag, arag

    # --- models
    from .models.embeddings.model import arun_embed, embedder, run_embed
    from .models.language.model import arun_llm, llm, run_llm


__all__ = [
    # --- core
    # zyx.core.schemas.schema
    "to_schema",
    # --- models
    # zyx.models.language.model
    "llm",
    "run_llm",
    "arun_llm",
    # zyx.models.embeddings.model
    "embedder",
    "run_embed",
    "arun_embed",
]


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
