"""zyx._strategies._attachments"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai import RunContext

from ._abstract import AbstractStrategy
from ..._processing._multimodal import (
    MultimodalContentOrigin,
    MultimodalContentMediaType,
    render_multimodal_source_as_text,
)
from ..._processing._toon import object_as_text


@dataclass
class AnchorEdit:
    start_anchor: str
    end_anchor: str
    replacement: str


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


@dataclass
class FileStrategy(AbstractStrategy):
    path: Path
    writeable: bool
    confirm: bool
    max_chars: int

    @property
    def kind(self) -> Literal["file"]:
        return "file"

    def get_description(self) -> str:
        return f"File at {self.path}"

    def get_state_description(self) -> str:
        origin = MultimodalContentOrigin.classify(self.path)
        media_type = MultimodalContentMediaType.classify(self.path, origin)
        text = render_multimodal_source_as_text(self.path, origin, media_type)
        if len(text) > self.max_chars:
            return text[: self.max_chars] + "\n\n[TRUNCATED]"
        return text

    def get_toolset(self) -> FunctionToolset | None:
        if not self.writeable:
            return None
        toolset = FunctionToolset()

        @toolset.tool
        def read_file() -> str:
            """Read the current contents of the file."""
            return self.get_state_description()

        @toolset.tool
        def write_file(
            edits: list[AnchorEdit] | None = None,
            content: str | None = None,
        ) -> str:
            """Apply anchor-based edits to the file, or replace it if content is provided."""
            if self.confirm:
                raise RuntimeError("File writes require confirmation.")
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

        @toolset.tool
        def append_file(content: str) -> str:
            """Append content to the file."""
            if self.confirm:
                raise RuntimeError("File writes require confirmation.")
            with self.path.open("a", encoding="utf-8") as f:
                f.write(content)
            return "ok"

        return toolset


@dataclass
class ObjectValueStrategy(AbstractStrategy):
    source: Any

    @property
    def kind(self) -> Literal["object"]:
        return "object"

    def get_description(self) -> str:
        return f"Object of type {type(self.source).__name__}"

    def get_state_description(self) -> str:
        return object_as_text(self.source)

    def get_toolset(self) -> FunctionToolset:
        toolset = FunctionToolset()

        @toolset.tool
        def read_object() -> str:
            """Read the current state of the attached object."""
            return self.get_state_description()

        @toolset.tool
        async def edit_object(ctx: RunContext[Any], instructions: str) -> str:
            """Edit the attached object using natural language instructions."""
            from ...operations.edit import aedit

            agent = None
            if hasattr(ctx.deps, "internal"):
                agent = ctx.deps.internal.get("agent")
                if agent is None:
                    agent = ctx.deps.internal.get("model")
            if agent is None:
                agent = getattr(ctx, "model", None)
            if agent is None:
                raise RuntimeError(
                    "No agent or model available in deps for edit_object."
                )

            model = agent
            if hasattr(agent, "model"):
                model = agent.model

            result = await aedit(
                target=self.source,
                context=instructions,
                model=model,
                selective=False,
                deps=getattr(ctx.deps, "user", ctx.deps),
            )
            if result.output is self.source:
                return self.get_state_description()
            if isinstance(self.source, dict) and isinstance(
                result.output, dict
            ):
                self.source.clear()
                self.source.update(result.output)
                return self.get_state_description()
            self.source = result.output
            return self.get_state_description()

        return toolset