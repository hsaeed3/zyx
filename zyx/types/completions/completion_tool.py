# zyx.types.completions.completion_tool
# tool base class


from pydantic import BaseModel
from typing import Any, Callable, Type, Dict, Optional, Union


class CompletionTool(BaseModel):
    """Internal tool class."""

    class Config:
        arbitrary_types_allowed = True

    # tool name (used for execution)
    name : Optional[str] = None

    # arguments (used for execution)
    arguments : Optional[Union[Dict[str, Any], Dict, Any]] = None

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

    is_string_tool : Optional[bool] = False