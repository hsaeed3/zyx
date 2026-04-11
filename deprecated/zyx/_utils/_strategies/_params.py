"""zyx._utils._strategies._params

Framework specific strategies for the `source` and
`target` parameters of semantic operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic_ai.messages import UserPromptPart

from ._abstract import AbstractStrategy
from ..._aliases import PydanticAIModelRequest
from ..._processing._multimodal import (
    MultimodalContentOrigin,
    MultimodalContentMediaType,
    render_multimodal_source_as_text,
    render_multimodal_source_as_user_content,
    render_multimodal_source_as_description,
)
from ..._processing._toon import object_as_text
from ...targets import Target
from ...attachments import Attachment


@dataclass
class SourcePayload:
    kind: Literal["raw", "multimodal"]
    content: str
    origin: MultimodalContentOrigin | None = None
    media_type: MultimodalContentMediaType | None = None
    source_repr: str | None = None
    message: PydanticAIModelRequest | None = None

    @property
    def is_text_based(self) -> bool:
        if self.media_type is None:
            return True
        return self.media_type in (
            MultimodalContentMediaType.TEXT,
            MultimodalContentMediaType.HTML,
            MultimodalContentMediaType.DOCUMENT,
            MultimodalContentMediaType.UNKNOWN,
        )


@dataclass
class SourceStrategy(AbstractStrategy):
    source: Any

    @property
    def kind(self) -> Literal["source"]:
        return "source"

    def get_payload(self) -> SourcePayload:
        if isinstance(self.source, Attachment):
            description = self.source.description
            content = self.source.text
            if content:
                if description:
                    content = f"{description}\n\n{content}"
                return SourcePayload(kind="raw", content=content)
            message = self.source.message
            return SourcePayload(
                kind="multimodal",
                content=description,
                message=message,
            )

        if isinstance(self.source, (str, Path)):
            path = Path(self.source)
            if path.exists() and path.is_file():
                origin = MultimodalContentOrigin.classify(path)
                media_type = MultimodalContentMediaType.classify(path, origin)
                is_text_based = media_type in (
                    MultimodalContentMediaType.TEXT,
                    MultimodalContentMediaType.HTML,
                    MultimodalContentMediaType.DOCUMENT,
                    MultimodalContentMediaType.UNKNOWN,
                )
                text = render_multimodal_source_as_text(
                    path, origin, media_type
                )
                description = render_multimodal_source_as_description(
                    path, origin, media_type
                )
                content = text if is_text_based else description
                message = None
                if not is_text_based:
                    user_content = render_multimodal_source_as_user_content(
                        path, origin, media_type
                    )
                    message = PydanticAIModelRequest(
                        parts=[UserPromptPart(content=[user_content])]
                    )
                return SourcePayload(
                    kind="multimodal",
                    content=content,
                    origin=origin,
                    media_type=media_type,
                    source_repr=str(path),
                    message=message,
                )

        return SourcePayload(
            kind="raw",
            content=object_as_text(self.source),
        )


@dataclass
class TargetStrategy(AbstractStrategy):
    target: Any
    name: str | None = None
    description: str | None = None
    instructions: Any | None = None
    has_hooks: bool = False
    is_target_wrapper: bool = False

    @property
    def kind(self) -> Literal["target"]:
        return "target"

    @classmethod
    def from_target(cls, target: Any) -> "TargetStrategy":
        if isinstance(target, Target):
            has_hooks = bool(
                target._field_hooks
                or target._prebuilt_hooks.get("complete", [])
            )
            return cls(
                target=target.target,
                name=target.name,
                description=target.description,
                instructions=target.instructions,
                has_hooks=has_hooks,
                is_target_wrapper=True,
            )
        return cls(target=target)
