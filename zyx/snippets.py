"""zyx.snippets"""

from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic_ai.messages import UserPromptPart

from ._aliases import PydanticAIModelRequest
from ._processing._multimodal import (
    MultimodalContentOrigin,
    MultimodalContentMediaType,
    render_multimodal_source_as_text,
    render_multimodal_source_as_user_content,
    render_multimodal_source_as_description,
)


SourceType = TypeVar("SourceType")


@dataclass(init=False)
class Snippet(Generic[SourceType]):
    """
    A `Snippet` is a piece of content that can be used to represent textual
    content, documents or multimodal (image, audio, video) content within
    the `context`, or attachments provided to agents and llms during an operation.

    This supports the following source types:

    - A `Path` object representing a local file
    - A string representing a URL to a text or multimodal file/html page
    - A bytes object representing raw binary data
    - An object representing a pydantic model or other object that can be
    encoded to text using the TOON format
    """

    def __init__(
        self,
        source: SourceType,
    ) -> None:
        self._source = source

        self._origin = MultimodalContentOrigin.classify(self._source)
        self._media_type = MultimodalContentMediaType.classify(
            self._source, self._origin
        )

    @property
    def source(self) -> SourceType:
        """
        The source input used to create this snippet.
        """
        return self._source

    @property
    def message(self) -> PydanticAIModelRequest:
        """
        Representation of this snippet as a `pydantic-ai` ModelRequest message.
        """
        content = render_multimodal_source_as_user_content(
            self._source,
            self._origin,
            self._media_type,
        )
        return PydanticAIModelRequest(
            parts=[UserPromptPart(content=[content])]
        )

    @property
    def text(self) -> str:
        """
        Renders this snippet as text, or a textual description in the case of multimodal
        content for this snippet.
        """
        return render_multimodal_source_as_text(
            self._source,
            self._origin,
            self._media_type,
        )

    @cached_property
    def description(self) -> str:
        """
        Renders this snippet as a description, or a textual description for
        multimodal content for this snippet.
        """
        return render_multimodal_source_as_description(
            self._source,
            self._origin,
            self._media_type,
        )

    def __rich__(self):
        from rich.console import RenderableType, Group
        from rich.rule import Rule
        from rich.text import Text

        renderables: list[RenderableType] = []
        renderables.append(
            Rule(title="✨ Snippet", style="rule.line", align="left")
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Source: {self._source!r}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Origin: {self._origin}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Media Type: {self._media_type}[/dim italic]"
            )
        )
        return Group(*renderables)


def paste(
    source: SourceType,
) -> Snippet[SourceType]:
    """
    Creates a new `Snippet` from a source object.

    A `Snippet` is a piece of content that can be used to represent textual
    content, documents or multimodal (image, audio, video) content within
    the `context`, or attachments provided to agents and llms during an operation.

    This supports the following source types:

    - A `Path` object representing a local file
    - A string representing a URL to a text or multimodal file/html page
    - A bytes object representing raw binary data
    - An object representing a pydantic model or other object that can be
    encoded to text using the TOON format

    Examples:
        >>> zyx.paste("pyproject.toml")
        >>> zyx.paste("https://www.google.com")
        >>> zyx.paste(bytes(b"Hello, world!"))
        >>> zyx.paste(json.dumps({"name": "John", "age": 30}))

    Args:
        source (SourceType): The source object to create a `Snippet` from.

    Returns:
        A new `Snippet` instance.
    """
    return Snippet(source)
