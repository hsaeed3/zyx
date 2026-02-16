"""zyx._graph._requests"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Generic, Type, TypeVar

from pydantic_ai.output import NativeOutput

from .._processing._toon import object_as_text
from .._processing._messages import (
    parse_instructions_as_system_prompt_parts,
)
from .._aliases import (
    PydanticAIModelRequest,
    PydanticAIMessage,
    PydanticAIToolset,
    PydanticAISystemPromptPart,
)
from .._utils._outputs import OutputBuilder
from .._utils._strategies._params import SourceStrategy, TargetStrategy
from ..attachments import Attachment
from ._ctx import (
    SemanticGraphContext,
)

__all__ = ("SemanticGraphRequestTemplate",)


Deps = TypeVar("Deps")
Output = TypeVar("Output")


@dataclass
class SemanticGraphRequestTemplate(Generic[Output]):
    """
    Template class used by various steps within a semantic operation's graph, to build
    the context of a single request.
    """

    output_type: Type[Output] | None = None
    """An override to the `output_type` parameter within a single step/agent run within a
    graph."""

    native_output: bool = False
    """Whether to use `pydantic_ai`'s `NativeOutput` type for the output of this request,
    this allows the model to use it's native structured output capabilities."""

    native_output_name: str | None = None
    """The name of the output to be used for this request."""

    native_output_description: str | None = None
    """The description of the output to be used for this request."""

    system_prompt_additions: str | list[str] | None = None
    """Dynamic instructions that can be applied onto the existing system prompt/instructions
    within `SemanticGraphDeps`. These are added before the primary input (source) block."""

    user_prompt_additions: str | list[str] | None = None
    """Dynamic user prompt content to append to the message history of a request."""

    toolsets: List[PydanticAIToolset] | None = None
    """Any operation-specific toolsets that can be included within the request."""

    include_run_context: bool = True
    """Whether to include the context of previous runs within the request."""

    include_source_context: bool = True
    """Whether to include context about a given `source` within the system prompt of a
    request. (If one is provided)"""

    include_output_context: bool = False
    """Whether to provide context about the current state of the output within the system
    prompt of a request."""

    include_user_toolsets: bool = True
    """Whether user-provided tools/toolsets should be included within this request."""

    def render(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> Dict[str, Any]:
        """
        Renders a dictionary of parameters compatible for invoking a `pydantic_ai.Agent` instance
        using the deps & state within a `SemanticGraphContext` object.
        """
        message_history = self._build_message_history(ctx)
        system_parts = self._build_system_prompt(ctx, message_history)
        if system_parts:
            message_history = [
                PydanticAIModelRequest(parts=system_parts),
                *message_history,
            ]

        self._apply_user_prompt_additions(message_history)
        toolsets = self._collect_toolsets(ctx)
        output_type = self._resolve_output_type(ctx)

        return {
            "message_history": message_history,
            "output_type": output_type,
            "toolsets": toolsets,
            "deps": ctx.deps.deps,
            "usage_limits": getattr(ctx.deps, "usage_limits", None),
        }

    def _build_message_history(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> List[PydanticAIMessage]:
        message_history: List[PydanticAIMessage] = (
            list(ctx.deps.message_history) if ctx.deps.message_history else []
        )

        if self.include_run_context and hasattr(ctx.state, "agent_runs"):
            for run in ctx.state.agent_runs:
                message_history.extend(run.new_messages())

        return message_history

    def _build_system_prompt(
        self,
        ctx: SemanticGraphContext[Output, Deps],
        message_history: List[PydanticAIMessage],
    ) -> List[PydanticAISystemPromptPart]:
        system_parts: List[PydanticAISystemPromptPart] = (
            list(ctx.deps.instructions) if ctx.deps.instructions else []
        )

        system_parts.extend(self._add_target_instructions(ctx))
        system_parts.extend(self._add_system_prompt_additions(ctx))
        system_parts.extend(self._add_attachments(ctx, message_history))

        if self.include_source_context:
            system_parts.extend(self._add_source_context(ctx, message_history))

        if (
            self.include_output_context
            and getattr(ctx.state, "output", None) is not None
        ):
            system_parts.append(
                PydanticAISystemPromptPart(
                    content=_render_output_context(ctx.state.output)
                )
            )

        return system_parts

    def _add_target_instructions(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> List[PydanticAISystemPromptPart]:
        target_strategy: TargetStrategy | None = getattr(
            ctx.deps, "target_strategy", None
        )
        if target_strategy is None:
            return []
        if not target_strategy.instructions:
            return []
        return parse_instructions_as_system_prompt_parts(
            instructions=target_strategy.instructions,
            deps=ctx.deps.deps,
        )

    def _add_system_prompt_additions(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> List[PydanticAISystemPromptPart]:
        if not self.system_prompt_additions:
            return []
        return parse_instructions_as_system_prompt_parts(
            instructions=self.system_prompt_additions,  # type: ignore
            deps=ctx.deps.deps,
        )

    def _add_attachments(
        self,
        ctx: SemanticGraphContext[Output, Deps],
        message_history: List[PydanticAIMessage],
    ) -> List[PydanticAISystemPromptPart]:
        parts: List[PydanticAISystemPromptPart] = []
        if not getattr(ctx.deps, "attachments", None):
            return parts

        for attachment in ctx.deps.attachments:  # type: ignore
            if isinstance(attachment, Attachment):
                parts.append(
                    PydanticAISystemPromptPart(
                        content=_render_attachment_context(
                            name=attachment.name
                            or type(attachment.source).__name__,
                            description=attachment.get_description(),
                            state=attachment.get_state_description(),
                        )
                    )
                )
                if attachment.message is not None:
                    if attachment.message not in message_history:
                        message_history.insert(0, attachment.message)
                continue

            parts.append(
                PydanticAISystemPromptPart(
                    content=_render_attachment_context(
                        name="Attachment",
                        description=str(attachment),
                        state=None,
                    )
                )
            )

        return parts

    def _add_source_context(
        self,
        ctx: SemanticGraphContext[Output, Deps],
        message_history: List[PydanticAIMessage],
    ) -> List[PydanticAISystemPromptPart]:
        parts: List[PydanticAISystemPromptPart] = []
        source_strategy: SourceStrategy | None = getattr(
            ctx.deps, "source_strategy", None
        )
        if source_strategy is None:
            if ctx.deps.source is None:
                return parts
            parts.append(
                PydanticAISystemPromptPart(
                    content=_render_source_context(ctx.deps.source)
                )
            )
            return parts

        payload = source_strategy.get_payload()
        if payload.kind == "raw":
            parts.append(
                PydanticAISystemPromptPart(
                    content=_render_source_context(payload.content)
                )
            )
            return parts

        parts.append(
            PydanticAISystemPromptPart(
                content=_render_source_metadata_values(
                    origin=payload.origin,
                    media_type=payload.media_type,
                    source_repr=payload.source_repr,
                )
            )
        )
        parts.append(
            PydanticAISystemPromptPart(
                content=_render_source_attachment_context(payload.content)
            )
        )
        if payload.message is not None:
            if payload.message not in message_history:
                message_history.insert(0, payload.message)
        return parts

    def _apply_user_prompt_additions(
        self, message_history: List[PydanticAIMessage]
    ) -> None:
        if not self.user_prompt_additions:
            return
        message_history.append(
            PydanticAIModelRequest.user_text_prompt(
                user_prompt=str(self.user_prompt_additions)
            )
        )

    def _collect_toolsets(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> List[PydanticAIToolset]:
        toolsets: List[PydanticAIToolset] = []
        if self.include_user_toolsets and ctx.deps.toolsets:
            toolsets.extend(ctx.deps.toolsets)
        if self.toolsets:
            toolsets.extend(self.toolsets)
        return toolsets

    def _resolve_output_type(
        self, ctx: SemanticGraphContext[Output, Deps]
    ) -> Type[Output] | None:
        target_strategy: TargetStrategy | None = getattr(
            ctx.deps, "target_strategy", None
        )
        if target_strategy is None:
            return self.output_type

        if not target_strategy.is_target_wrapper:
            output_type = self.output_type
            if (
                output_type is None
                and getattr(ctx.state, "output", None) is not None
            ):
                output_type = getattr(ctx.state.output, "normalized", None)

            if self.native_output:
                if output_type is not str:
                    output_type = NativeOutput(
                        outputs=output_type,
                        name=target_strategy.name or self.native_output_name,
                        description=target_strategy.description
                        or self.native_output_description,
                    )
            return output_type  # type: ignore

        output_type = NativeOutput(
            outputs=ctx.state.output.normalized,
            name=target_strategy.name,
            description=target_strategy.description,
        )
        if target_strategy.has_hooks:
            return None

        return output_type  # type: ignore


def _render_output_context(builder: OutputBuilder[Output]) -> str:
    """Generates system prompt context for the current state of the output
    builder.
    """
    if builder.is_value:
        context = (
            f"\n\n[Output Context]\n"
            f"You are currently building/generating an output of type: {object_as_text(builder.normalized)}\n"
            f"The output had a starting state of: {object_as_text(builder.target)}\n"
        )
    else:
        context = (
            f"\n\n[Output Context]\n"
            f"You are currently building/generating an output of type: {object_as_text(builder.normalized)}\n"
        )

    if builder.partial is not None:
        context += f"The current state of the output is: {object_as_text(builder.partial)}\n"

    return context


def _render_source_context(source: Any | Type[Any]) -> str:
    """Generates system prompt context for a source object or value.

    The only content between the markers is the raw source; no instructional
    line inside the block, so the model returns exactly that content when
    asked to extract the primary input.
    """
    body = object_as_text(source)
    return f"\n\n[PRIMARY INPUT]\n\n{body}\n\n[END PRIMARY INPUT]\n"


def _render_source_attachment_context(content: str) -> str:
    """Generates system prompt context for a source attachment."""
    return f"\n\n[PRIMARY INPUT]\n\n{content}\n\n[END PRIMARY INPUT]\n"


def _render_source_metadata_values(
    *, origin: Any, media_type: Any, source_repr: str | None
) -> str:
    """Generates system prompt metadata for a source."""
    origin_value = getattr(origin, "value", origin)
    media_value = getattr(media_type, "value", media_type)
    source_value = source_repr or ""
    return (
        "\n\n[SOURCE META]\n"
        f"Origin: {origin_value}\n"
        f"Media Type: {media_value}\n"
        f"Source: {source_value}\n"
        "[END SOURCE META]\n"
    )


def _render_attachment_context(
    name: str | None,
    description: str,
    state: str | None,
) -> str:
    title = name or "Attachment"
    body = f"\n\n[ATTACHMENT: {title}]\n"
    body += f"Description:\n{description}\n"
    if state:
        body += f"\nState:\n{state}\n"
    body += "[END ATTACHMENT]\n"
    return body
