"""zyx.result"""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

__all__ = ("Result",)


T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """
    Standardized representation of a result generated from a
    'generative operation'.

    This class is generic to type `T`, of the `output`, which represents
    the target set during invocation.
    """

    output: T | None = Field(default=None)
    """The 'final'/primary output of this result."""
