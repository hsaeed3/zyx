"""zyx.targets"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Generic,
    Type,
    TypeAlias,
    TypeVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from ._types import ModelParam


Output = TypeVar("Output")


TargetHook: TypeAlias = Literal["complete", "error"]


@dataclass
class Target(Generic[Output]):
    """
    A structured way to define the `target` or output to generate
    through a semantic operation.
    """

    target: Output | Type[Output]
    """The target type or value that will be generated/mutated by
    semantic operations."""

    name: str | None = field(default=None)
    """An optional human readable name for this target type, this is useful
    for targets of primitive types as well as for better model understanding
    of a target."""

    description: str | None = field(default=None)
    """An optional human readable description of this target type."""

    instructions: str | None = field(default=None)
    """Specific instructions for the model to follow when generating this target."""

    constraints: List[str] | None = field(default=None)
    """An optional list of constraints that this target type must adhere to,
    these can be explicitly validated by a model if needed."""

    model: ModelParam | None = field(default=None)
    """Default model/pydantic ai agent to use for operations on this target (e.g. ``validate``)."""

    _field_hooks: Dict[str, Callable[[Any], Any]] = field(
        init=False, default_factory=dict
    )
    """A dictionary of field hooks that can be used to validate/mutate this target
    as it is being generated/mutated.

    You can set a field hook on a target by using the `@on_field` decorator."""

    _prebuilt_hooks: Dict[TargetHook, List[Callable[[Any], Any]]] = field(
        init=False, default_factory=dict
    )
    """A list of prebuilt hooks that can be used to validate/mutate this target
    based on a certain event or condition.

    You can apply a prebuilt hook on a target by using the `@on` decorator."""

    def on_field(self, field: str | None = None):
        """
        Decorator to register a 'field hook' on this target. A field hook
        is a function that can be used to validate/mutate a specific field
        on the event it is generated or mutated by a model.

        For targets with types such as `str`, `int`, with no field names, you
        can leave the `field` argument as None.
        """

        def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
            key = field or "__self__"
            self._field_hooks.setdefault(key, []).append(fn)  # type: ignore
            return fn

        return decorator

    def on(self, event: TargetHook):
        """
        Decorator to register a 'prebuilt hook' on this target.

        Hooks include:
        - `complete`: Called when the target is generated/mutated successfully.
        - `error`: Called when the target is generated/mutated with an error.
        """

        def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
            self._prebuilt_hooks.setdefault(event, []).append(fn)
            return fn

        return decorator
