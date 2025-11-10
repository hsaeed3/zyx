"""zyx.core.processing.text"""

from .chunking import needs_chunking, to_chunks
from .ingestion import (
    identify_string_kind,
    ingest_file_as_string,
    ingest_url_as_string,
    string_has_template_syntax,
)
from .templating import render_template_string

__all__ = (
    "to_chunks",
    "needs_chunking",
    "identify_string_kind",
    "ingest_file_as_string",
    "ingest_url_as_string",
    "string_has_template_syntax",
    "render_template_string",
)
