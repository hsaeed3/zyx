"""zyx.context"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Generic, List, TypeVar, TYPE_CHECKING

from pydantic_ai import messages as _messages
from pydantic_ai.toolsets import FunctionToolset

from ._aliases import (
    PydanticAIInstructions,
    PydanticAIMessage,
    PydanticAIModelRequest,
    PydanticAIModelResponse,
    PydanticAIAgentResult,
    PydanticAIBuiltinTool,
    PydanticAITool,
    PydanticAIToolset,
    PydanticAIUserContent,
)
from ._processing._messages import parse_context_to_pydantic_ai_messages

if TYPE_CHECKING:
    from ._types import ToolType, ContextType


Deps = TypeVar("Deps")


@dataclass(init=False)
class Context(Generic[Deps]):
    """Simple and mutable container for the conversation/message history across
    multiple semantic operations, with the optional ability to set context-specific
    instructions, tools and `deps` that can be passed to `pydantic_ai`'s RunContext."""

    _messages: List[PydanticAIMessage] = field(default_factory=list)

    instructions: str | Callable[[Deps | None], str] | None = None
    """System-level instructions, either a string or a callable. If callable,
    it may optionally accept `deps` and should return a string."""

    tools: List[ToolType] = field(default_factory=list)
    """Tools to include in the context. This can be a sequence of functions, pydantic ai tools, builtin tools
    or toolsets."""

    deps: Deps | None = None
    """`deps` that can be passed to messages, tools and instructions through `pydantic_ai`'s
    RunContext."""

    update: bool = True
    """When `True` (default), semantic operations will automatically update this context after their result
    is completed."""

    compact_instructions: bool = False
    """When `True`, all instructions are compacted into a single `SystemPromptPart`; otherwise they are split on double newlines."""

    exclude_messages: bool = False
    """When `True`, existing messages are not forwarded to operations."""

    exclude_instructions: bool = False
    """When `True`, instructions are not rendered into messages."""

    exclude_tools: bool = False
    """When `True`, tools are not rendered into messages."""

    max_length: int | None = None
    """Optional cap on the number of messages forwarded, from the end of the message history.
    This does not affect the inclusion of instructions."""

    def __init__(
        self,
        messages: ContextType | List[ContextType] | None = None,
        instructions: PydanticAIInstructions | None = None,
        tools: ToolType | List[ToolType] | None = None,
        deps: Deps | None = None,
        update: bool = True,
        compact_instructions: bool = False,
        exclude_messages: bool = False,
        exclude_instructions: bool = False,
        exclude_tools: bool = False,
        max_length: int | None = None,
    ) -> None:
        """
        Initializes a new `Context` object with the given parameters.

        Args:
            messages : Sequence[Any] | None = None
                Initial messages to seed the context. This is any value accepted by
                `parse_context_to_pydantic_ai_messages`.
            instructions : str | Callable[[Deps | None], str] | None = None
                System-level instructions, either a string or a callable. If callable,
                it may optionally accept `deps` and should return a string.
            tools : Sequence[ToolType] | ToolType | None = None
                Tools to include in the context. This can be a sequence of functions, pydantic ai tools, builtin tools
                or toolsets.
            deps : Deps | None = None
                Reference to `deps` in `pydantic_ai.RunContext`, that can be passed to messages,
                tools and instructions.
            update : bool = True
                When `True` (default), semantic operations that consume this context will write back the full message history.
            compact_instructions : bool = False
                When `True`, all instructions are compacted into a single `SystemPromptPart`; otherwise they are split on double newlines.
            exclude_messages : bool = False
                When `True`, existing messages are not forwarded to operations.
            exclude_instructions : bool = False
                When `True`, instructions are not rendered into messages.
            exclude_tools : bool = False
                When `True`, tools are not rendered into messages.
            max_length : int | None = None
                Optional cap on the number of messages forwarded (from the tail).
        """
        if messages:
            self._messages = parse_context_to_pydantic_ai_messages(messages)
        else:
            self._messages = []

        self.instructions = instructions  # type: ignore

        if tools:
            if not isinstance(tools, list):
                tools = [tools]
            self.tools = tools  # type: ignore

        self.deps = deps
        self.update = update
        self.compact_instructions = compact_instructions
        self.exclude_messages = exclude_messages
        self.exclude_instructions = exclude_instructions
        self.exclude_tools = exclude_tools
        self.max_length = max_length

    def __call__(
        self,
        *,
        instructions: PydanticAIInstructions | None = None,
        tools: ToolType | List[ToolType] | None = None,
        deps: Deps | None = None,
        update: bool | None = None,
        compact_instructions: bool | None = None,
        exclude_messages: bool | None = None,
        exclude_instructions: bool | None = None,
        exclude_tools: bool | None = None,
        max_length: int | None = None,
    ) -> Context[Deps]:
        """Update the content and configuration of this `Context` with per-call overrides and
        return a new `Context` with the updated content.

        NOTE: This method creates a copy of this `Context` object before updates
        are applied, so no updates are set to the original `Context` object.

        Args:
            instructions : PydanticAIInstructions | None = None
                System-level instructions, either a string or a callable. If callable,
                it may optionally accept `deps` and should return a string.
            tools : ToolType | List[ToolType] | None = None
                Tools to include in the context. This can be a sequence of functions, pydantic ai tools, builtin tools
                or toolsets.
            deps : Deps | None = None
                Reference to `deps` in `pydantic_ai.RunContext`, that can be passed to messages,
                tools and instructions.
            update : bool | None = None
                When `True` (default), semantic operations that consume this context will write back the full message history.
            compact_instructions : bool | None = None
                When `True`, all instructions are compacted into a single `SystemPromptPart`; otherwise they are split on double newlines.
            exclude_messages : bool | None = False
                When `True`, existing messages are not forwarded to operations.
            exclude_instructions : bool | None = False
                When `True`, instructions are not rendered into messages.
            exclude_tools : bool | None = False
                When `True`, tools are not rendered into messages.
            max_length : int | None = None
                Optional cap on the number of messages forwarded, from the end of the message history.
                This does not affect the inclusion of instructions.
        """
        cls = self.copy()

        if instructions is not None:
            cls.instructions = instructions  # type: ignore
        if tools is not None:
            if not isinstance(tools, list):
                tools = [tools]
            cls.tools = tools  # type: ignore
        if deps is not None:
            cls.deps = deps
        if update is not None:
            cls.update = update
        if compact_instructions is not None:
            cls.compact_instructions = compact_instructions
        if exclude_messages is not None:
            cls.exclude_messages = exclude_messages
        if exclude_instructions is not None:
            cls.exclude_instructions = exclude_instructions
        if exclude_tools is not None:
            cls.exclude_tools = exclude_tools
        if max_length is not None:
            cls.max_length = max_length

        return cls

    def copy(self) -> Context[Deps]:
        cls = Context(
            messages=None,
            instructions=self.instructions,
            tools=self.tools,
            deps=self.deps,
            update=self.update,
            compact_instructions=self.compact_instructions,
            exclude_messages=self.exclude_messages,
            exclude_instructions=self.exclude_instructions,
            exclude_tools=self.exclude_tools,
            max_length=self.max_length,
        )
        cls._messages = self._messages

        return cls

    def clear(self) -> None:
        """Reset the message history of this `Context` to an empty list."""
        self._messages.clear()

    def add_user_message(
        self,
        content: str | List[PydanticAIUserContent],
        index: int | None = None,
    ) -> None:
        """
        Append or inject a user message to the message history if `index`
        is given.

        Args:
            content : str | List[PydanticAIUserContent]
                The content of the user message. This can be a string or a list of `PydanticAIUserContent` objects.
            index : int | None = None
                The index at which to inject the user message. If not given,
                the message is appended to the end of the message history.
        """
        if index is not None:
            self._messages.insert(
                index,
                PydanticAIModelRequest(
                    parts=[_messages.UserPromptPart(content=content)]
                ),
            )
        else:
            self._messages.append(
                PydanticAIModelRequest(
                    parts=[_messages.UserPromptPart(content=content)]
                )
            )

    def add_assistant_message(
        self,
        content: str,
        index: int | None = None,
    ) -> None:
        """
        Append or inject an assistant message to the message history if `index`
        is given.

        Args:
            content : str
                The content of the assistant message.
            index : int | None = None
                The index at which to inject the assistant message. If not given,
                the message is appended to the end of the message history.
        """
        if index is not None:
            self._messages.insert(
                index,
                PydanticAIModelResponse(
                    parts=[_messages.TextPart(content=content)]
                ),
            )
        else:
            self._messages.append(
                PydanticAIModelResponse(
                    parts=[_messages.TextPart(content=content)]
                )
            )

    def add_system_message(
        self,
        content: str,
        index: int | None = None,
    ) -> None:
        """
        Append or inject a system message to the message history if `index`
        is given.

        Args:
            content : str
                The content of the system message.
            index : int | None = None
                The index at which to inject the system message. If not given,
                the message is appended to the end of the message history.
        """
        if index is not None:
            self._messages.insert(
                index,
                PydanticAIModelRequest(
                    parts=[_messages.SystemPromptPart(content=content)]
                ),
            )
        else:
            self._messages.append(
                PydanticAIModelRequest(
                    parts=[_messages.SystemPromptPart(content=content)]
                )
            )

    def render_messages(self) -> List[PydanticAIMessage]:
        """Renders the message history of this `Context` into a list of `PydanticAIMessage` objects."""
        if self.exclude_messages:
            return []

        messages = list(self._messages)

        if self.max_length is not None and self.max_length >= 0:
            if len(messages) > self.max_length:
                messages = messages[-self.max_length :]

        return messages

    def render_instructions(self) -> List[_messages.SystemPromptPart]:  # type: ignore
        """Render instructions into one or more ``SystemPromptPart`` objects."""
        if self.exclude_instructions or not self.instructions:
            return []

        instruction = self.instructions
        resolved: Any

        if callable(instruction):
            fn_sig = inspect.signature(instruction)
            params = fn_sig.parameters
            should_pass_deps = False

            if params:
                first_param = next(iter(params.values()))
                if first_param.annotation is not inspect._empty:
                    if self.deps is not None and isinstance(
                        self.deps, first_param.annotation
                    ):
                        should_pass_deps = True
                elif len(params) == 1:
                    should_pass_deps = True

            try:
                if should_pass_deps:
                    resolved = instruction(self.deps)
                else:
                    resolved = instruction()
            except Exception:
                resolved = (
                    instruction(self.deps)
                    if self.deps is not None
                    else instruction()
                )
        else:
            resolved = instruction

        content = resolved.strip() if isinstance(resolved, str) else None
        if not content:
            return []

        if self.compact_instructions:
            return [_messages.SystemPromptPart(content=content)]

        parts = [s for s in content.split("\n\n") if s.strip()]
        return [_messages.SystemPromptPart(content=part) for part in parts]

    def render_toolsets(self) -> List[PydanticAIToolset]:
        """Render the toolsets of this `Context` into a list of `PydanticAIToolset` objects."""
        if self.exclude_tools:
            return []

        toolsets = []
        function_tools = []

        for tool in self.tools:
            if inspect.isfunction(tool):
                function_tools.append(PydanticAITool(function=tool))
            elif isinstance(tool, (PydanticAITool, PydanticAIBuiltinTool)):
                function_tools.append(tool)
            elif isinstance(tool, PydanticAIToolset):
                toolsets.append(tool)
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")

        if function_tools:
            toolsets.append(FunctionToolset(tools=function_tools))

        return toolsets

    def update_from_pydantic_ai_result(
        self, result: PydanticAIAgentResult[Any]
    ) -> None:
        """Update the message history of this `Context` with the full exchange from a run."""
        try:
            self._messages.extend(result.new_messages())
        except Exception:
            pass

    def __rich__(self):
        from rich.console import RenderableType, Group
        from rich.rule import Rule
        from rich.text import Text

        renderables: list[RenderableType] = []

        renderables.append(
            Rule(title="✨ Context", style="rule.line", align="left")
        )

        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Messages: {len(self._messages)}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Instructions: {self.instructions!r}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Tools: {len(self.tools)}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Deps: {bool(self.deps)}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Update on Result: {self.update}[/dim italic]"
            )
        )

        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Compact Instructions: {self.compact_instructions}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Exclude Messages: {self.exclude_messages}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Exclude Instructions: {self.exclude_instructions}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Exclude Tools: {self.exclude_tools}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Max Length: {self.max_length}[/dim italic]"
            )
        )

        return Group(*renderables)


def create_context(
    *,
    messages: ContextType | List[ContextType] | None = None,
    instructions: PydanticAIInstructions | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    update: bool = True,
    compact_instructions: bool = False,
    exclude_messages: bool = False,
    exclude_instructions: bool = False,
    exclude_tools: bool = False,
    max_length: int | None = None,
) -> Context[Deps]:
    """Create a new `Context` object with the given parameters.

    Args:
        messages : Sequence[Any] | None = None
            Initial messages to seed the context. This is any value accepted by
            `parse_context_to_pydantic_ai_messages`.
        instructions : str | Callable[[Deps | None], str] | None = None
            System-level instructions, either a string or a callable. If callable,
            it may optionally accept `deps` and should return a string.
        tools : Sequence[ToolType] | ToolType | None = None
            Tools to include in the context. This can be a sequence of functions, pydantic ai tools, builtin tools
            or toolsets.
        deps : Deps | None = None
            Reference to `deps` in `pydantic_ai.RunContext`, that can be passed to messages,
            tools and instructions.
        update : bool = True
            When `True` (default), semantic operations that consume this context will write back the full message history.
        compact_instructions : bool = False
            When `True`, all instructions are compacted into a single `SystemPromptPart`; otherwise they are split on double newlines.
        exclude_messages : bool = False
            When `True`, existing messages are not forwarded to operations.
        exclude_instructions : bool = False
            When `True`, instructions are not rendered into messages.
        exclude_tools : bool = False
            When `True`, tools are not rendered into messages.
        max_length : int | None = None
            Optional cap on the number of messages forwarded (from the tail).

    Returns:
        Context[Deps]
            A new `Context` object with the given parameters.
    """
    return Context(
        messages=messages,
        instructions=instructions,
        tools=tools,
        deps=deps,
        update=update,
        compact_instructions=compact_instructions,
        exclude_messages=exclude_messages,
        exclude_instructions=exclude_instructions,
        exclude_tools=exclude_tools,
        max_length=max_length,
    )
