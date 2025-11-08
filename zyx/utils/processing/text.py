"""zyx.utils.processing.text

God bless everyone who made these following utils:"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Iterable
from functools import lru_cache

from ...core.exceptions import ProcessingError

if TYPE_CHECKING:
    from markitdown import MarkItDown
    from chonkie.chunker.base import BaseChunker, Chunk
    from chonkie.chunker.recursive import RecursiveChunker


@lru_cache(maxsize=1)
def cached_markitdown_instance() -> MarkItDown:
    """Lazily load and cache the default MarkItDown instance."""
    from markitdown import MarkItDown

    return MarkItDown()


@lru_cache(maxsize=1)
def cached_recursive_chunker() -> RecursiveChunker:
    """Lazily load and cache the default RecursiveChunker."""
    import logging
    from chonkie.chunker.recursive import RecursiveChunker

    # Suppress chonkie logs (DEBUG and INFO)
    logging.getLogger("chonkie").setLevel(logging.ERROR)

    return RecursiveChunker(chunk_size=2048)


def read_file_as_markdown(file: str) -> str:
    """Read a file and convert its content to markdown using MarkItDown."""
    markitdown = cached_markitdown_instance()
    try:
        return markitdown.convert_local(file).markdown
    except Exception as e:
        raise ProcessingError(
            f"Failed to convert file to markdown: {e}", "TEXT:MARKDOWN"
        ) from e


def read_url_as_markdown(url: str) -> str:
    """Read a URL and convert its content to markdown using MarkItDown."""
    markitdown = cached_markitdown_instance()
    try:
        return markitdown.convert_url(url).markdown
    except Exception as e:
        raise ProcessingError(
            f"Failed to convert URL to markdown: {e}", "TEXT:MARKDOWN"
        ) from e


def chunk_text(
    text: str,
    chunker: BaseChunker,
) -> Iterable[Chunk]:
    """Chunk text using a given chunker."""
    try:
        return chunker.chunk(text)
    except Exception as e:
        raise ProcessingError(f"Failed to chunk text: {e}", "TEXT:CHUNKING") from e
