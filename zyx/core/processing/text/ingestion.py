"""zyx.utils.processing.text.ingestion"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from httpx import URL

if TYPE_CHECKING:
    from markitdown import MarkItDown

__all__ = [
    "is_ingestible_string",
    "ingest_url_as_string",
    "ingest_file_as_string",
    "string_has_template_syntax",
]


StringKind: TypeAlias = Literal[
    "url", "file:simple", "file:complex", "string"
]
"""The kind of ingestible content an input string is."""


@lru_cache(maxsize=1)
def _get_markitdown_instance() -> MarkItDown:
    """Retrieves a singleton instance of a MarkItDown
    client."""
    from markitdown import DocumentConverterResult, MarkItDown

    return MarkItDown()


def identify_string_kind(string: str | Path | URL) -> str:
    """Identify the kind of ingestible content an input string is.
    file, URL or just a simple string. Ingestible content in the
    context of `zyx` is any string that points to
    external content that can be 'ingested' as a string."""

    @lru_cache(maxsize=1000)
    def _is_ingestible_string(s: str | Path | URL) -> bool:
        # Check if it's a Path object or can be converted to one
        if isinstance(s, Path):
            return s.exists() and s.is_file()

        # Check if it's a URL object
        if isinstance(s, URL):
            return "url"

        # Convert to string for further checks
        string_val = str(s)

        # Check if it's a URL (http/https)
        if string_val.startswith(("http://", "https://")):
            return "url"

        # Check if it's a file path that exists
        try:
            path = Path(string_val)

            if path.suffix in (".markdown", ".md", ".txt"):
                tag = "file:simple"
            else:
                tag = "file:complex"

            if path.exists() and path.is_file():
                return tag
        except (OSError, ValueError):
            # Invalid path or permission error
            pass

        # It's just a regular string
        return "string"

    return _is_ingestible_string(string)


@lru_cache(maxsize=1000)
def string_has_template_syntax(string: str) -> bool:
    """Check if a string contains Jinja2-style template syntax.

    Detects the presence of variable interpolation ({{ }}) or
    control structures ({%  %}) commonly used in template engines.
    Uses efficient string searching to minimize overhead.

    Args:
        string: The string to check for template syntax

    Returns:
        True if template syntax is detected, False otherwise
    """
    if not string:
        return False

    return "{{" in string or "{%" in string or "{#" in string


def ingest_url_as_string(url: str) -> str:
    """Uses MarkItDown to ingest a URLs content as a markdown
    string."""

    @lru_cache(maxsize=1000)
    def _ingest_url_as_string(u: URL) -> str:
        """Uses MarkItDown to ingest a URL's content as a markdown
        string."""
        content = _get_markitdown_instance().convert_url(u)
        if content.title:
            return f"# {content.title}\n\n{content.markdown}"
        return content.markdown

    if isinstance(url, URL):
        # markitdown uses .strip(), which is not a behvaior of
        # httpx.URL
        url = url.__str__()

    return _ingest_url_as_string(url)


@lru_cache(maxsize=1000)
def ingest_simple_document_as_string(document: str) -> str:
    """Uses MarkItDown to ingest a simple document's content as a markdown
    string.

    A 'simple' document is:
    - .markdown
    - .md
    - .txt
    """
    if not document.endswith((".markdown", ".md", ".txt")):
        raise ValueError("Document must be a .markdown, .md or .txt file")

    return Path(document).read_text()


def ingest_file_as_string(document: Path | URL | str) -> str:
    """Uses MarkItDown to ingest a document's content as a markdown
    string."""

    if document.endswith((".markdown", ".md", ".txt")):
        return ingest_simple_document_as_string(document)

    @lru_cache(maxsize=1000)
    def _ingest_file_as_string(d: Path | URL) -> str:
        """Uses MarkItDown to ingest a document's content as a markdown
        string."""
        content = _get_markitdown_instance().convert_local(d)

        if content.title:
            return f"# {content.title}\n\n{content.markdown}"
        return content.markdown

    return _ingest_file_as_string(document)
