"""zyx.ai

Primary namespace for all interfaces and utilities directly in the context
of interacting with and generating content from AI models.

This is one of the lowest level namepsaces in the library, and implements
simple wrappers over various AI model `adapters` or backend clients, and
with the `instructor` library for using language models to generate structured
outputs; along with various utility classes and resources for working with
generative AI related aspects such as function calling.
"""

from ..core.utils._import_utils import *

if TYPE_CHECKING:
    from .models.adapters import ModelAdapter
    from .models.providers import ModelProvider, ModelProviderInfo

    from .models.language.model import LanguageModel, llm, allm
    from .models.language.types import (
        LanguageModelName,
        LanguageModelSettings,
        LanguageModelResponse,
    )

    from .models.embeddings.model import EmbeddingModel, embed, aembed
    from .models.embeddings.types import (
        EmbeddingModelName,
        EmbeddingModelSettings,
        EmbeddingModelResponse,
    )

    from .utils.function_calling import openai_function_schema
    from .utils.structured_outputs import prepare_structured_output_model


__all__ = [
    "ModelAdapter",
    "ModelProvider",
    "ModelProviderInfo",
    "LanguageModel",
    "llm",
    "allm",
    "LanguageModelName",
    "LanguageModelSettings",
    "LanguageModelResponse",
    "EmbeddingModel",
    "embed",
    "aembed",
    "EmbeddingModelName",
    "EmbeddingModelSettings",
    "EmbeddingModelResponse",
    "openai_function_schema",
    "prepare_structured_output_model",
]


__getattr__ = type_checking_getattr_fn(__all__)
__dir__ = type_checking_dir_fn(__all__)
