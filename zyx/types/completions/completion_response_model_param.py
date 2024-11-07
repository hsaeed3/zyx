# zyx.types.completions.completion_response_model_param
# response model types for completions

from pydantic import BaseModel
from typing import Any, Dict, Type, Union


# response model
CompletionResponseModelParam = Union[
    # standard response_model input
    BaseModel,
    Type[BaseModel],

    # quick string response (not a type -> the string becomes the field name)
    str,

    # dict -- converted to a pydantic model
    Dict[str, Any],

    # standard types -- all converted to simple pydantic models
    Type[int], Type[float], Type[str], Type[bool], Type[list]
]