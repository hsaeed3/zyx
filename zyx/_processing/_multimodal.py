"""zyx.processing.multimodal"""

from __future__ import annotations

from functools import lru_cache
from enum import Enum
from io import BytesIO
import mimetypes
import re
from pathlib import Path
from typing import Any, TypeAlias, Self, Tuple, TYPE_CHECKING

from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    VideoUrl,
    UserContent,
)

from ._toon import object_as_toon_text

if TYPE_CHECKING:
    from markitdown import MarkItDown


MultimodalContent: TypeAlias = Path | str | bytes | Any
"""
The source location from where a piece of external content can originate from.

This can be one of the following types:
- A `Path` object representing a local file
- A string representing a URL to a text or multimodal file/html page
- A bytes object representing raw binary data
- Any non-string python object that can be encoded to text using the TOON format
"""


_URIS = ("http://", "https://", "ftp://")
_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    (".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".ini", ".toml", ".log")
)
_DOCUMENT_MIMETYPES: frozenset[str] = frozenset(
    (
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
)


class MultimodalContentOrigin(Enum):
    """
    The computed origin of a snippet.
    """

    FILE = "file"
    URL = "url"
    STRING = "string"
    BYTES = "bytes"
    OBJECT = "object"

    @classmethod
    def classify(cls, source: MultimodalContent) -> Self:
        """
        Classifies the the origin of a snippet's source.
        """
        if isinstance(source, (bytes, bytearray, memoryview)):
            return cls.BYTES  # type: ignore

        elif isinstance(source, Path):
            if source.is_file():
                return cls.FILE  # type: ignore

        else:
            if isinstance(source, str):
                if source.startswith(_URIS):
                    return cls.URL  # type: ignore
                if Path(source).exists() and Path(source).is_file():
                    return cls.FILE  # type: ignore

            else:
                return cls.OBJECT  # type: ignore

        return cls.STRING  # type: ignore


class MultimodalContentMediaType(Enum):
    """
    The media/content type of a source.
    """

    TEXT = "text/plain"
    HTML = "text/html"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"

    @classmethod
    def classify(
        cls, source: MultimodalContent, origin: MultimodalContentOrigin
    ) -> Self:
        """
        Classifies the the type of a snippet's content.
        """
        if (
            origin == MultimodalContentOrigin.OBJECT
            or origin == MultimodalContentOrigin.STRING
        ):
            # objects are encoded to text using the TOON format
            return cls.TEXT  # type: ignore

        if origin == MultimodalContentOrigin.BYTES:
            data = (
                source.tobytes()
                if isinstance(source, memoryview)
                else bytes(source)  # type: ignore
            )
            head = data[:512]
            if head.startswith(b"%PDF"):
                return cls.DOCUMENT  # type: ignore
            if (
                head.startswith(b"\x89PNG\r\n\x1a\n")
                or head[:3] == b"\xff\xd8\xff"
            ):
                return cls.IMAGE  # type: ignore
            if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
                return cls.IMAGE  # type: ignore
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                return cls.IMAGE  # type: ignore
            if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
                return cls.AUDIO  # type: ignore
            if (
                head.startswith(b"OggS")
                or head.startswith(b"fLaC")
                or head.startswith(b"ID3")
            ):
                return cls.AUDIO  # type: ignore
            if len(head) >= 8 and head[4:8] == b"ftyp":
                return cls.VIDEO  # type: ignore
            if head.startswith(b"\x1a\x45\xdf\xa3"):
                return cls.VIDEO  # type: ignore
            if b"\x00" in head:
                return cls.UNKNOWN  # type: ignore
            try:
                text = head.decode("utf-8")
            except UnicodeDecodeError:
                return cls.UNKNOWN  # type: ignore
            if re.search(
                r"(?i)<!doctype html|<html\\b|<head\\b|<body\\b", text
            ):
                return cls.HTML  # type: ignore
            return cls.TEXT  # type: ignore

        path_text = ""
        suffix = ""
        if origin == MultimodalContentOrigin.FILE:
            path = source if isinstance(source, Path) else Path(source)  # type: ignore
            path_text = str(path)
            suffix = path.suffix.lower()
        elif origin == MultimodalContentOrigin.URL and isinstance(source, str):
            path_text = re.split(r"[?#]", source, 1)[0]
            suffix = Path(path_text).suffix.lower()

        if suffix in _TEXT_EXTENSIONS:
            return cls.TEXT  # type: ignore
        if suffix in (".html", ".htm", ".xhtml"):
            return cls.HTML  # type: ignore
        if suffix in (
            ".pdf",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
        ):
            return cls.DOCUMENT  # type: ignore

        mime_type, _ = mimetypes.guess_type(path_text)
        if mime_type == "text/html":
            return cls.HTML  # type: ignore
        if mime_type in _DOCUMENT_MIMETYPES:
            return cls.DOCUMENT  # type: ignore
        if mime_type:
            if mime_type.startswith("image/"):
                return cls.IMAGE  # type: ignore
            if mime_type.startswith("audio/"):
                return cls.AUDIO  # type: ignore
            if mime_type.startswith("video/"):
                return cls.VIDEO  # type: ignore
            if mime_type.startswith("text/"):
                return cls.TEXT  # type: ignore

        return cls.UNKNOWN  # type: ignore


def classify_multimodal_source(
    source: MultimodalContent,
) -> Tuple[MultimodalContentOrigin, MultimodalContentMediaType]:
    """
    Classifies a multimodal source into its origin and type.
    """
    origin = MultimodalContentOrigin.classify(source)
    return origin, MultimodalContentMediaType.classify(source, origin)


@lru_cache(maxsize=1)
def _get_markdown_converter() -> MarkItDown:
    from markitdown import MarkItDown

    return MarkItDown()


def _render_multimodal_file_or_url_as_text(source: str) -> str:
    try:
        return _get_markdown_converter().convert(source).markdown
    except Exception as e:
        raise ValueError(f"Failed to render file or URL as text: {e}") from e


def render_multimodal_source_as_user_content(
    source: MultimodalContent,
    origin: MultimodalContentOrigin,
    media_type: MultimodalContentMediaType,
) -> UserContent:
    """
    Renders a multimodal source as a compatible `pydantic-ai` UserContent object.
    """
    if origin == MultimodalContentOrigin.OBJECT:
        return object_as_toon_text(source)

    if origin == MultimodalContentOrigin.STRING:
        if media_type == MultimodalContentMediaType.HTML:
            from markitdown import StreamInfo

            buffer = BytesIO(str(source).encode("utf-8"))
            info = StreamInfo(mimetype="text/html", extension=".html")
            return (
                _get_markdown_converter()
                .convert(buffer, stream_info=info)
                .markdown
            )
        return str(source)

    if origin == MultimodalContentOrigin.URL and isinstance(source, str):
        guessed_media_type, _ = mimetypes.guess_type(source)
        if media_type == MultimodalContentMediaType.IMAGE:
            return ImageUrl(url=source, media_type=guessed_media_type)
        if media_type == MultimodalContentMediaType.AUDIO:
            return AudioUrl(url=source, media_type=guessed_media_type)
        if media_type == MultimodalContentMediaType.VIDEO:
            return VideoUrl(url=source, media_type=guessed_media_type)
        if media_type in (
            MultimodalContentMediaType.TEXT,
            MultimodalContentMediaType.HTML,
            MultimodalContentMediaType.DOCUMENT,
        ):
            return _render_multimodal_file_or_url_as_text(source)
        return (
            DocumentUrl(url=source, media_type=guessed_media_type)
            if guessed_media_type
            else source
        )

    if origin == MultimodalContentOrigin.FILE:
        path = source if isinstance(source, Path) else Path(source)  # type: ignore
        if media_type in (
            MultimodalContentMediaType.TEXT,
            MultimodalContentMediaType.HTML,
            MultimodalContentMediaType.DOCUMENT,
        ):
            return _render_multimodal_file_or_url_as_text(str(path))
        guessed_media_type, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        return BinaryContent.narrow_type(
            BinaryContent(
                data=data,
                media_type=guessed_media_type or "application/octet-stream",
            )
        )

    if origin == MultimodalContentOrigin.BYTES:
        data = (
            source.tobytes()
            if isinstance(source, memoryview)
            else bytes(source)  # type: ignore
        )
        if media_type == MultimodalContentMediaType.HTML:
            from markitdown import StreamInfo

            buffer = BytesIO(data)
            info = StreamInfo(mimetype="text/html", extension=".html")
            return (
                _get_markdown_converter()
                .convert(buffer, stream_info=info)
                .markdown
            )
        if media_type == MultimodalContentMediaType.TEXT:
            return data.decode("utf-8")
        return BinaryContent.narrow_type(
            BinaryContent(data=data, media_type="application/octet-stream")
        )

    return str(source)


def render_multimodal_source_as_text(
    source: MultimodalContent,
    origin: MultimodalContentOrigin,
    media_type: MultimodalContentMediaType,
) -> str:
    """
    Renders a multimodal source as text, or a textual description for
    multimodal content.
    """
    if origin == MultimodalContentOrigin.OBJECT:
        return object_as_toon_text(source)

    if media_type in (
        MultimodalContentMediaType.TEXT,
        MultimodalContentMediaType.HTML,
        MultimodalContentMediaType.DOCUMENT,
    ):
        if origin == MultimodalContentOrigin.STRING and isinstance(
            source, str
        ):
            return source

        if origin == MultimodalContentOrigin.FILE:
            path = source if isinstance(source, Path) else Path(source)  # type: ignore
            return _render_multimodal_file_or_url_as_text(str(path))

        if origin == MultimodalContentOrigin.URL and isinstance(source, str):
            return _render_multimodal_file_or_url_as_text(source)

        if origin == MultimodalContentOrigin.BYTES:
            data = (
                source.tobytes()
                if isinstance(source, memoryview)
                else bytes(source)  # type: ignore
            )
            if media_type == MultimodalContentMediaType.TEXT:
                return data.decode("utf-8")
            if media_type == MultimodalContentMediaType.HTML:
                from markitdown import StreamInfo

                buffer = BytesIO(data)
                info = StreamInfo(mimetype="text/html", extension=".html")
                return (
                    _get_markdown_converter()
                    .convert(buffer, stream_info=info)
                    .markdown
                )
            return "[document bytes]"

    if origin == MultimodalContentOrigin.URL and isinstance(source, str):
        return f"[{media_type.value} content at {source}]"
    if origin == MultimodalContentOrigin.FILE:
        path = source if isinstance(source, Path) else Path(source)  # type: ignore
        return f"[{media_type.value} content from {path}]"
    if origin == MultimodalContentOrigin.BYTES:
        return f"[{media_type.value} content bytes]"

    return str(source)


def render_multimodal_source_as_description(
    source: MultimodalContent,
    origin: MultimodalContentOrigin,
    media_type: MultimodalContentMediaType,
) -> str:
    """
    Renders a multimodal source as a description, or a textual description for
    multimodal content.
    """
    if origin == MultimodalContentOrigin.OBJECT:
        return f"[object of type {type(source).__name__}]"

    if origin == MultimodalContentOrigin.STRING:
        length = len(source) if isinstance(source, str) else 0
        return f"[string value, length={length}]"

    if origin == MultimodalContentOrigin.URL:
        return f"[remote {media_type.value} resource]"

    if origin == MultimodalContentOrigin.FILE:
        path = source if isinstance(source, Path) else Path(source)  # type: ignore[arg-type]
        name = path.name or "file"
        return f"[{media_type.value} file: {name}]"

    if origin == MultimodalContentOrigin.BYTES:
        size = (
            len(source)
            if isinstance(source, (bytes, bytearray))
            else len(bytes(source))  # type: ignore
        )
        return f"[{media_type.value} bytes, size={size}]"

    return f"[{media_type.value} content]"
