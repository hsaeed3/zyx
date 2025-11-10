"""zyx.core"""

from typing import TYPE_CHECKING

from .._internal import _import_utils

if TYPE_CHECKING:
    from .interfaces.bits import Bit, to_bit
    from .interfaces.maker import Maker, aedit, amake, edit, make
    from .interfaces.stuff import Stuff, to_stuff
    from .interfaces.things import Thing, to_thing
    from .models.clients import ModelClient
    from .models.definition import ModelDefinition
    from .models.embeddings.model import (
        ChonkieEmbeddingModel,
        EmbeddingModel,
        EmbeddingModelResponse,
    )
    from .models.language.model import LanguageModel
    from .models.language.types import LanguageModelResponse
    from .models.providers import (
        ModelProvider,
        ModelProviderName,
        ModelProviderRegistry,
    )
    from .processing.schemas.openai import to_openai_schema
    from .processing.schemas.pydantic import to_pydantic_model
    from .processing.schemas.schema import Schema, to_schema
    from .processing.text.text import Text, to_text


__all__ = (
    # zyx.core.interfaces.maker
    "Maker",
    "aedit",
    "amake",
    "edit",
    "make",
    # zyx.core.interfaces.bits
    "Bit",
    "to_bit",
    # zyx.core.interfaces.things
    "Thing",
    "to_thing",
    # zyx.core.interfaces.stuff
    "Stuff",
    "to_stuff",
    # zyx.core.models.clients
    "ModelClient",
    # zyx.core.models.definition
    "ModelDefinition",
    # zyx.core.models.providers
    "ModelProvider",
    "ModelProviderName",
    "ModelProviderRegistry",
    # zyx.core.models.embeddings.model
    "EmbeddingModel",
    "ChonkieEmbeddingModel",
    "EmbeddingModelResponse",
    # zyx.core.models.language.model
    "LanguageModel",
    "LanguageModelResponse",
    # zyx.core.processing.schemas.openai
    "to_openai_schema",
    # zyx.core.processing.schemas.pydantic
    "to_pydantic_model",
    # zyx.core.processing.schemas.schema
    "Schema",
    "to_schema",
    # zyx.core.processing.schemas.semantics
    "to_semantic_title",
    "to_semantic_key",
    "to_semantic_description",
    # zyx.core.processing.text.text
    "Text",
    "to_text",
)


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
