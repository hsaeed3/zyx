"""⚡️ zyx"""

from typing import TYPE_CHECKING

from ._internal import _import_utils

if TYPE_CHECKING:
    from .core.interfaces.bits import Bit, to_bit
    from .core.interfaces.maker import Maker as maker
    from .core.interfaces.maker import aedit, amake, edit, make
    from .core.interfaces.stuff import Stuff, to_stuff
    from .core.interfaces.things import Thing, to_thing
    from .core.models.embeddings.model import (
        ChonkieEmbeddingModel as chonkie_embedder,
    )
    from .core.models.embeddings.model import EmbeddingModel as embedder
    from .core.models.language.model import LanguageModel as llm
    from .core.processing.schemas.openai import (
        to_openai_schema as to_openai_schema,
    )
    from .core.processing.schemas.pydantic import to_pydantic_model
    from .core.processing.schemas.schema import to_schema as schema
    from .core.processing.text.text import to_text as text


__all__ = (
    # zyx.core.interfaces.bits
    "Bit",
    "to_bit",
    # zyx.core.interfaces.things
    "Thing",
    "to_thing",
    # zyx.core.interfaces.maker
    "maker",
    "aedit",
    "edit",
    "amake",
    "make",
    # zyx.core.interfaces.stuff
    "Stuff",
    "to_stuff",
    # zyx.core.models.language.model
    "llm",
    # zyx.core.models.embeddings.model
    "embedder",
    "chonkie_embedder",
    # zyx.core.processing.schemas.schema
    "schema",
    # zyx.core.processing.schemas.openai
    "to_openai_schema",
    # zyx.core.processing.schemas.pydantic
    "to_pydantic_model",
    # zyx.core.processing.text.text
    "text",
)


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
