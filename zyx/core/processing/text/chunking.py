"""zyx.utils.processing.text.chunking"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List

from ...._internal._exceptions import ProcessingError

if TYPE_CHECKING:
    from chonkie.chunker.base import BaseChunker
    from chonkie.chunker.recursive import RecursiveChunker
    from chonkie.types import Chunk

__all__ = [
    "needs_chunking",
    "to_chunks",
]


@lru_cache(maxsize=1)
def _get_chunker() -> RecursiveChunker:
    """Returns a singleton instance of a chonkie 'RecursiveChunker',
    used as the library default within `zyx`.
    """
    from chonkie.chunker.recursive import RecursiveChunker

    return RecursiveChunker()


def needs_chunking(text: str, chunk_size: int = 2048) -> bool:
    """Simple boolean check to see if a text string needs to be chunked."""
    return len(text) > chunk_size


def to_chunks(
    text: str, chunk_size: int = 2048, chunker: BaseChunker | None = None
) -> List[Chunk]:
    """Chunks a text string using either a provided chonkie chunker,
    or the library default."""

    @lru_cache(maxsize=1000)
    def _chunk_text(t: str, c: BaseChunker) -> List[Chunk]:
        """Chunks a text string using a provided chonkie chunker."""
        c.chunk_size = chunk_size

        return c.chunk(t)

    if chunker is None:
        chunker = _get_chunker()

    try:
        return _chunk_text(text, chunker)
    except Exception as e:
        raise ProcessingError(f"Failed to chunk text: {e}") from e
