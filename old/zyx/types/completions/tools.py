# zyx.types.completions.tools
# tool typing & base class


from pydantic import BaseModel
from typing import Any, Callable, Type, Dict, Optional, Union


ToolType = Union[str, Callable, Type[BaseModel], Dict[str, Any]]


class Tool(BaseModel):
    """Internal tool class."""

    class Config:
        arbitrary_types_allowed = True

    # tool name (used for execution)
    name : Optional[str] = None

    # arguments (used for execution)
    arguments : Optional[Dict[str, Any]] = None

    # description
    description : Optional[str] = None

    # function
    function : Union[
        # openai function
        Dict[str, Any],
        # pydantic model
        Type[BaseModel],
        # callable
        Callable
    ]

    # formatted function (completion request)
    formatted_function : Optional[Dict[str, Any]] = None