"""zyx.core.processing"""

from typing import TYPE_CHECKING

from ..._internal import _import_utils

if TYPE_CHECKING:
    from .schemas import (
        Schema,
        to_openai_schema,
        to_pydantic_model,
        to_schema,
        to_semantic_description,
        to_semantic_key,
        to_semantic_title,
        validate_with_pydantic_model,
    )
    from .text import (
        identify_string_kind,
        ingest_file_as_string,
        ingest_url_as_string,
        needs_chunking,
        render_template_string,
        string_has_template_syntax,
        to_chunks,
    )


__all__ = (
    # zyx.core.processing.text.chunking
    "to_chunks",
    "needs_chunking",
    # zyx.core.processing.text.ingestion
    "ingest_file_as_string",
    "ingest_url_as_string",
    "identify_string_kind",
    "string_has_template_syntax",
    # zyx.core.processing.text.templating
    "render_template_string",
    # zyx.core.processing.schemas.schema
    "to_schema",
    "Schema",
    # zyx.core.processing.schemas.openai
    "to_openai_schema",
    # zyx.core.processing.schemas.pydantic
    "to_pydantic_model",
    "validate_with_pydantic_model",
    # zyx.core.processing.schemas.semantics
    "to_semantic_title",
    "to_semantic_key",
    "to_semantic_description",
)


__getattr__ = _import_utils.type_checking_getattr_fn(__all__)
__dir__ = _import_utils.type_checking_dir_fn(__all__)
