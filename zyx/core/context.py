"""zyx.context"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, TypedDict, cast
from uuid import uuid4


class Message(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | List[Any] | None
    name: str
    tool_calls: List[Any]
    tool_call_id: str


_active_context: ContextVar[Context | None] = cast(
    ContextVar["Context | None"],
    ContextVar("_active_context", default=None),
)


@dataclass
class Context:
    """
    A container for message history that can be passed to `make()` and `run()`.

    Manages conversation state in OpenAI message format internally, allowing
    seamless multi-turn interactions.

    Examples
    --------
    >>> ctx = zyx.context()
    >>> ctx.add("user", "Hello!")
    >>> response = zyx.make(ctx, target=str)

    >>> # As context manager (auto-tracks responses)
    >>> with zyx.context() as ctx:
    ...     response = zyx.make("What is 2+2?", target=int)
    ...     followup = zyx.make("Double that", target=int)

    >>> # With system prompt
    >>> ctx = zyx.context(system="You are a helpful assistant")
    """

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    """Unique identifier for this context."""

    messages: List[Message] = field(default_factory=list)
    """The message history in OpenAI format."""

    system: str | None = None
    """Optional system prompt prepended to messages."""

    def __post_init__(self) -> None:
        if self.system:
            self.messages.insert(
                0, {"role": "system", "content": self.system}
            )

    def __enter__(self) -> Context:
        self._token = _active_context.set(self)
        return self

    def __exit__(self, *_: Any) -> None:
        _active_context.reset(self._token)

    def __iter__(self):
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def add(
        self,
        role: Literal["system", "user", "assistant", "tool"],
        content: str | List[Any] | None = None,
        **kwargs: Any,
    ) -> Context:
        """Add a message to the context. Returns self for chaining."""
        msg: Message = {"role": role, "content": content, **kwargs}  # type: ignore
        self.messages.append(msg)
        return self

    def user(self, content: str | List[Any]) -> Context:
        """Add a user message."""
        return self.add("user", content)

    def assistant(
        self, content: str | None = None, **kwargs: Any
    ) -> Context:
        """Add an assistant message (with optional tool_calls)."""
        return self.add("assistant", content, **kwargs)

    def tool(self, tool_call_id: str, name: str, content: str) -> Context:
        """Add a tool result message."""
        return self.add(
            "tool", content, tool_call_id=tool_call_id, name=name
        )

    def fork(self) -> Context:
        """Create a copy of this context for branching conversations."""
        return Context(
            messages=[m.copy() for m in self.messages],
            system=self.system,
        )

    def clear(self) -> Context:
        """Clear all messages (keeps system prompt if set)."""
        self.messages.clear()
        if self.system:
            self.messages.append(
                {"role": "system", "content": self.system}
            )
        return self

    def dump(self) -> List[Message]:
        """Return a copy of messages for API consumption."""
        return list(self.messages)

    @property
    def last(self) -> Message | None:
        """The most recent message, or None."""
        return self.messages[-1] if self.messages else None


def resolve_context(
    context: List[Dict[str, Any] | Context | str] | Context | str,
) -> List[Dict[str, Any]]:
    """
    Resolve various context input types into a flat list of messages.
    """
    messages: List[Dict[str, Any]] = []

    # Check for active context first
    active_ctx = get_active_context()
    if active_ctx is not None and context is not active_ctx:
        messages.extend(active_ctx.dump())

    if isinstance(context, str):
        messages.append({"role": "user", "content": context})
    elif isinstance(context, Context):
        messages.extend(context.dump())
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, Context):
                # Skip if this is the active context (already added above)
                if item is not active_ctx:
                    messages.extend(item.dump())
            elif isinstance(item, dict):
                messages.append(item)
    else:
        raise TypeError(f"Invalid context type: {type(context)}")

    return messages


def context(
    system: str | None = None,
    messages: List[Message] | None = None,
) -> Context:
    """
    Create a new conversation context.

    Parameters
    ----------
    system : str, optional
        System prompt to prepend to all interactions.
    messages : List[Message], optional
        Initial messages to populate the context with.

    Returns
    -------
    Context
        A new context instance.
    """
    ctx = Context(system=system)
    if messages:
        ctx.messages.extend(messages)
    return ctx


def get_active_context() -> Context | None:
    """Get the currently active context (if inside a `with context():` block)."""
    return _active_context.get()
