"""zyx.attachments"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeGuard, runtime_checkable
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.messages import UserPromptPart

from ._aliases import PydanticAIModelRequest, PydanticAIToolset
from ._processing._multimodal import (
    MultimodalContentOrigin,
    MultimodalContentMediaType,
    render_multimodal_source_as_text,
    render_multimodal_source_as_user_content,
    render_multimodal_source_as_description,
)
from ._utils._strategies._attachments import (
    FileStrategy,
    ObjectValueStrategy,
)

__all__ = (
    "Attachment",
    "AttachmentLike",
    "attach",
    "is_attachment_like",
    "paste",
)


class Attachment:
    """
    An `Attachment` is defined best as 2 sources of content:

    1. A piece of content that can be `pasted` to represent textual
    content, documents or multimodal (image, audio, video) content within
    the `context`, or attachments provided to agents and llms during an operation.

    2. A piece of content that can be `attached` to represent a
    pythonic object or external source that can be queried/edited/etc. by
    an agent or model.

    This supports the following source types:

    - A `Path` object representing a local file
    - A string representing a URL to a text or multimodal file/html page
    - A bytes object representing raw binary data
    - An object representing a pydantic model or other object that can be
    encoded to text using the TOON format
    - And much more...
    """

    def __init__(
        self,
        source: Any,
        *,
        name: str | None = None,
        interactive: bool = False,
        writeable: bool = False,
        confirm: bool = True,
        max_chars: int = 8000,
    ) -> None:
        self.source = source
        self.name = name
        self.interactive = interactive
        self._origin: MultimodalContentOrigin | None = None
        self._media_type: MultimodalContentMediaType | None = None

        self._origin = MultimodalContentOrigin.classify(self.source)
        self._media_type = MultimodalContentMediaType.classify(
            self.source, self._origin
        )
        path = Path(source) if isinstance(source, (str, Path)) else None
        if path is not None and path.exists() and path.is_file():
            self._strategy = FileStrategy(
                path=path,
                writeable=interactive or writeable,
                confirm=confirm,
                max_chars=max_chars,
            )
        else:
            self._strategy = ObjectValueStrategy(source)

    def get_description(self) -> str:
        return self._strategy.get_description()

    def get_state_description(self) -> str | None:
        return self._strategy.get_state_description()

    def get_toolset(self) -> FunctionToolset | None:
        if not self.interactive:
            return None
        return self._strategy.get_toolset()

    @property
    def message(self) -> PydanticAIModelRequest | None:
        if self._origin is None or self._media_type is None:
            return None
        content = render_multimodal_source_as_user_content(
            self.source,
            self._origin,
            self._media_type,
        )
        return PydanticAIModelRequest(
            parts=[UserPromptPart(content=[content])]
        )

    @property
    def description(self) -> str:
        if self._origin is None or self._media_type is None:
            return self.get_description()
        return render_multimodal_source_as_description(
            self.source,
            self._origin,
            self._media_type,
        )

    @property
    def text(self) -> str:
        if self._origin is None or self._media_type is None:
            return self.get_state_description() or ""
        return render_multimodal_source_as_text(
            self.source,
            self._origin,
            self._media_type,
        )


@runtime_checkable
class AttachmentLike(Protocol):
    def get_description(self) -> str: ...

    def get_state_description(self) -> str | None: ...

    def get_toolset(self) -> PydanticAIToolset | None: ...

    @property
    def message(self) -> PydanticAIModelRequest | None: ...

    @property
    def name(self) -> str | None: ...


def is_attachment_like(value: Any) -> TypeGuard[AttachmentLike]:
    return (
        hasattr(value, "get_description")
        and hasattr(value, "get_state_description")
        and hasattr(value, "get_toolset")
    )


def attach(
    source: Any,
    *,
    name: str | None = None,
    writeable: bool = True,
    confirm: bool = True,
    max_chars: int = 8000,
) -> Attachment:
    """
    Attach an attachment from a source object or reference.

    This creates an interactive interface in which a model can use to query, mutate and otherwise
    interact with the attachment.

    Args:
        source (Any): The source object or reference to attach.
        name (str | None): The name of the attachment. Defaults to None.
        writeable (bool): Whether the attachment is writeable. Defaults to True.
        confirm (bool): Whether to confirm the attachment. Defaults to True.
        max_chars (int): The maximum number of characters to attach. Defaults to 8000.

    Returns:
        Attachment: The attachment object.
    """
    return Attachment(
        source,
        name=name,
        interactive=True,
        writeable=writeable,
        confirm=confirm,
        max_chars=max_chars,
    )


def paste(
    source: Any,
    *,
    name: str | None = None,
    max_chars: int = 8000,
) -> Attachment:
    """
    'Paste' an attachment from a source object or reference.

    A source can include a variety of data types, including:
    - A string representing a URL to a web page, multimodal content such as PDFs, images, audio, video, etc.
    - A local file path to a file on the filesystem
    - A bytes object representing raw binary data
    - A Pythonic object that can be encoded to text using the TOON format
    - And more...

    Args:
        source (Any): The source object or reference to paste.
        name (str | None): The name of the attachment. Defaults to None.
        max_chars (int): The maximum number of characters to paste. Defaults to 8000.

    Examples:
    ```python
    from zyx import paste
    attachment = paste("https://zyx.hammad.app")
    print(attachment.text)
    ```

    Returns:
        Attachment: The attachment object.
    """
    return Attachment(
        source,
        name=name,
        interactive=False,
        max_chars=max_chars,
    )
