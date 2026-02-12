"""zyx._graph._requests"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Generic, Type, TypeVar

from pydantic_ai.output import NativeOutput

from .._processing._toon import object_as_toon_text
from .._processing._messages import (
    parse_instructions_as_system_prompt_parts,
)
from .._processing._multimodal import MultimodalContentMediaType
from .._aliases import (
    PydanticAIModelRequest,
    PydanticAIMessage,
    PydanticAIToolset,
    PydanticAISystemPromptPart,
)
from .._utils._outputs import OutputBuilder
from ..resources.abstract import AbstractResource
from ..snippets import Snippet
from ._context import (
    SemanticGraphContext,
)


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
        params: Dict[str, Any] = {}

        # Start with base message history
        message_history: List[PydanticAIMessage] = (
            list(ctx.deps.message_history) if ctx.deps.message_history else []
        )

        # Optionally include run context
        if self.include_run_context and hasattr(ctx.state, "agent_runs"):
            for run in ctx.state.agent_runs:
                message_history.extend(run.new_messages())

        # Initialize system instructions (system prompt parts)
        system_parts: List[PydanticAISystemPromptPart] = (
            list(ctx.deps.instructions) if ctx.deps.instructions else []
        )

        # If the Target exists and has instructions, add those as well
        target = getattr(ctx.deps, "target", None)
        from ..targets import Target as TargetClass

        if target is not None and isinstance(target, TargetClass):
            if getattr(target, "instructions", None):
                system_parts.extend(
                    parse_instructions_as_system_prompt_parts(
                        instructions=target.instructions,
                        deps=ctx.deps.deps,
                    )
                )

        # System prompt additions (before source so instructions are seen first)
        if self.system_prompt_additions:
            system_parts.extend(
                parse_instructions_as_system_prompt_parts(
                    instructions=self.system_prompt_additions,  # type: ignore
                    deps=ctx.deps.deps,
                )
            )

        # Attachments
        if getattr(ctx.deps, "attachments", None):
            for attachment in ctx.deps.attachments:  # type: ignore
                if isinstance(attachment, Snippet):
                    system_parts.append(
                        PydanticAISystemPromptPart(
                            content=_render_attachment_context(
                                name="Snippet",
                                description=attachment.description,
                                state=None,
                            )
                        )
                    )

                    if attachment.message not in message_history:
                        message_history.insert(0, attachment.message)
                elif isinstance(attachment, AbstractResource):
                    system_parts.append(
                        PydanticAISystemPromptPart(
                            content=_render_attachment_context(
                                name=attachment.name,
                                description=attachment.get_description(),
                                state=attachment.get_state_description(),
                            )
                        )
                    )
                else:
                    system_parts.append(
                        PydanticAISystemPromptPart(
                            content=_render_attachment_context(
                                name="Attachment",
                                description=str(attachment),
                                state=None,
                            )
                        )
                    )

        if self.include_source_context and ctx.deps.source is not None:
            if isinstance(ctx.deps.source, AbstractResource):
                if ctx.deps.source.__class__.__name__ != "File":
                    raise ValueError(
                        f"Invalid Resource passed as `source` parameter: {ctx.deps.source.__class__.__name__}. "
                        "Currently only `File` resources can be used as the `source` of a semantic operation."
                    )
                snippet = Snippet(source=ctx.deps.source.path)
            elif isinstance(ctx.deps.source, Snippet):
                snippet = ctx.deps.source
            else:
                system_parts.append(
                    PydanticAISystemPromptPart(
                        content=_render_source_context(ctx.deps.source)
                    )
                )
                snippet = None

            if snippet is not None:
                is_text_based = snippet._media_type in (
                    MultimodalContentMediaType.TEXT,
                    MultimodalContentMediaType.HTML,
                    MultimodalContentMediaType.DOCUMENT,
                    MultimodalContentMediaType.UNKNOWN,
                )

                content = (
                    snippet.text if is_text_based else snippet.description
                )
                system_parts.append(
                    PydanticAISystemPromptPart(
                        content=_render_source_attachment_context(content)
                    )
                )

                if not is_text_based:
                    snippet_message = snippet.message
                    if snippet_message not in message_history:
                        message_history.insert(0, snippet_message)

        # Output context
        if (
            self.include_output_context
            and getattr(ctx.state, "output", None) is not None
        ):
            system_parts.append(
                PydanticAISystemPromptPart(
                    content=_render_output_context(ctx.state.output)
                )
            )

        # If we have system instructions, prepend a ModelRequest with those as system prompt parts
        if system_parts:
            message_history = [
                PydanticAIModelRequest(parts=system_parts),
                *message_history,
            ]

        # Optional user prompt additions
        if self.user_prompt_additions:
            message_history.append(
                PydanticAIModelRequest.user_text_prompt(
                    user_prompt=str(self.user_prompt_additions)
                )
            )

        # Compose toolsets: include user toolsets? add explicit toolsets?
        toolsets: List[PydanticAIToolset] = []
        if self.include_user_toolsets and ctx.deps.toolsets:
            toolsets.extend(ctx.deps.toolsets)
        if self.toolsets:
            toolsets.extend(self.toolsets)

        if not isinstance(ctx.deps.target, TargetClass):
            output_type = self.output_type
            if (
                output_type is None
                and getattr(ctx.state, "output", None) is not None
            ):
                output_type = getattr(ctx.state.output, "normalized", None)

            if self.native_output:
                output_type = NativeOutput(
                    outputs=output_type,
                    name=self.native_output_name
                    if self.native_output_name
                    else None,
                    description=self.native_output_description
                    if self.native_output_description
                    else None,
                )
        else:
            output_type = NativeOutput(
                outputs=ctx.state.output.normalized,
                name=ctx.deps.target.name,
                description=ctx.deps.target.description,
            )
            if (
                ctx.deps.target._field_hooks
                or ctx.deps.target._prebuilt_hooks.get("complete", [])
            ):
                output_type = None

        params = {
            "message_history": message_history,
            "output_type": output_type,
            "toolsets": toolsets,
            "deps": ctx.deps.deps,
            "usage_limits": getattr(ctx.deps, "usage_limits", None),
        }

        return params


def _render_output_context(builder: OutputBuilder[Output]) -> str:
    """Generates system prompt context for the current state of the output
    builder.
    """
    if builder.is_value:
        context = (
            f"\n\n[Output Context]\n"
            f"You are currently building/generating an output of type: {object_as_toon_text(builder.normalized)}\n"
            f"The output had a starting state of: {object_as_toon_text(builder.target)}\n"
        )
    else:
        context = (
            f"\n\n[Output Context]\n"
            f"You are currently building/generating an output of type: {object_as_toon_text(builder.normalized)}\n"
        )

    if builder.partial is not None:
        context += f"The current state of the output is: {object_as_toon_text(builder.partial)}\n"

    return context


def _render_source_context(source: Any | Type[Any]) -> str:
    """Generates system prompt context for a source object or value.

    The only content between the markers is the raw source; no instructional
    line inside the block, so the model returns exactly that content when
    asked to extract the primary input.
    """
    body = object_as_toon_text(source)
    return f"\n\n[PRIMARY INPUT]\n\n{body}\n\n[END PRIMARY INPUT]\n"


def _render_source_attachment_context(content: str) -> str:
    """Generates system prompt context for a source attachment (e.g. Snippet).

    For text-based content, the full text appears between the markers.
    For non-text content (images/audio/video), only the description appears.
    """
    return f"\n\n[PRIMARY INPUT]\n\n{content}\n\n[END PRIMARY INPUT]\n"


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
