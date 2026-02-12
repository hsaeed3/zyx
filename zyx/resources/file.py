"""zyx.resources.file"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, List

from pydantic import BaseModel, Field
from pydantic_ai.toolsets import FunctionToolset

from .._processing._multimodal import (
    MultimodalContentOrigin,
    MultimodalContentMediaType,
    render_multimodal_source_as_text,
    render_multimodal_source_as_description,
)
from .abstract import AbstractResource


_FILE_LOCKS: dict[str, threading.RLock] = {}


@dataclass(init=False)
class File(AbstractResource):
    """
    A resource that represents a text/JSON/YAML/TOML/INI file on the filesystem.
    """

    path: Path
    max_chars: int

    def __init__(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        writeable: bool = False,
        confirm: bool = True,
        max_chars: int = 8000,
    ) -> None:
        self.path = Path(path)
        resource_name = name or self.path.name
        super().__init__(
            name=resource_name, writeable=writeable, confirm=confirm
        )
        self.max_chars = max_chars

    def _ensure_exists(self) -> None:
        if not self.path.exists() or not self.path.is_file():
            raise FileNotFoundError(f"File not found: {self.path}")

    def _read_text(self) -> str:
        self._ensure_exists()
        origin = MultimodalContentOrigin.classify(self.path)
        media_type = MultimodalContentMediaType.classify(self.path, origin)
        text = render_multimodal_source_as_text(self.path, origin, media_type)
        if len(text) > self.max_chars:
            return text[: self.max_chars] + "\n\n[TRUNCATED]"
        return text

    def _describe(self) -> str:
        origin = MultimodalContentOrigin.classify(self.path)
        media_type = MultimodalContentMediaType.classify(self.path, origin)
        description = render_multimodal_source_as_description(
            self.path, origin, media_type
        )
        return description

    def get_description(self) -> str:
        return f"File at {self.path}"

    def get_state_description(self) -> str:
        with _get_lock(self.path):
            return self._describe()

    def get_toolset(self) -> FunctionToolset:
        toolset = FunctionToolset()

        @toolset.tool
        def read_file() -> str:
            """Read the current contents of the file."""
            with _get_lock(self.path):
                return self._read_text()

        if self.writeable:

            @toolset.tool
            def write_file(
                edits: List["AnchorEdit"] | None = None,
                content: str | None = None,
            ) -> str:
                """Apply anchor-based edits to the file, or replace the file if content is provided."""
                if self.confirm:
                    raise RuntimeError("File writes require confirmation.")
                with _get_lock(self.path):
                    self._ensure_exists()
                    if edits:
                        text = self.path.read_text(encoding="utf-8")
                        updated = text
                        for edit in edits:
                            updated = _apply_anchor_edit(
                                updated,
                                start_anchor=edit.start_anchor,
                                end_anchor=edit.end_anchor,
                                replacement=edit.replacement,
                            )
                        self.path.write_text(updated, encoding="utf-8")
                        return "ok"
                    if content is not None:
                        self.path.write_text(content, encoding="utf-8")
                        return "ok"
                return "ok"

            @toolset.tool
            def append_file(content: str) -> str:
                """Append content to the file."""
                if self.confirm:
                    raise RuntimeError("File writes require confirmation.")
                with _get_lock(self.path):
                    self._ensure_exists()
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(content)
                return "ok"

        return toolset


class AnchorEdit(BaseModel):
    """A single anchor-based edit operation for file content."""

    start_anchor: str = Field(
        ...,
        description=(
            "The starting text/content of the section that is being edited."
        ),
    )
    end_anchor: str = Field(
        ...,
        description=(
            "The ending text/content of the section that is being edited."
        ),
    )
    replacement: str = Field(
        ...,
        description="The text/content to replace the section with.",
    )


def _apply_anchor_edit(
    text: str, *, start_anchor: str, end_anchor: str, replacement: str
) -> str:
    start_idx = text.find(start_anchor)
    if start_idx == -1:
        return text
    end_idx = text.find(end_anchor, start_idx + len(start_anchor))
    if end_idx == -1:
        return text
    end_idx += len(end_anchor)
    return text[:start_idx] + replacement + text[end_idx:]


def _get_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    lock = _FILE_LOCKS.get(key)
    if lock is None:
        lock = threading.RLock()
        _FILE_LOCKS[key] = lock
    return lock
