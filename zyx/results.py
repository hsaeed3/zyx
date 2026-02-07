"""zyx.results"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Literal,
    Generic,
    TypeAlias,
    TypeVar,
)

from pydantic_ai.run import AgentRunResult

from .utils.prettify import prettify_result


Output = TypeVar("Output")


SemanticOperationKind : TypeAlias = Literal[
    "make",
]
"""
Alias representation of the various semantic operation kinds provided
by `zyx`.
"""


@dataclass(frozen=True)
class Result(Generic[Output]):
    """
    Standardized result object for most semantic operations.

    A `Result` is generic to type `Output` which represents the final output
    of the semantic operation.
    """

    kind : SemanticOperationKind = field(repr=False)
    """
    The kind of result this `Result` represents. (what semantic operation produced
    this result).
    """

    output : Output | None = None
    """
    The final output of the semantic operation.

    This is either a complete/partial representation of the `target` type or
    value set during invocation.
    """

    raw : AgentRunResult[Output] | None = field(
        default = None,
        repr = False
    )
    """
    The raw result from the underlying `Pydantic AI` agent run(s) used to
    produce the `output` value. In some cases, an output is not a result
    of a single agent run.
    """

    models : list[str] = field(
        default_factory = list,
    )
    """The model(s) used to produce the `output` value."""

    def __rich__(self):
        return prettify_result(self)