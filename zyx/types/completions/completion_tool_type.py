# zyx.types.completions.completion_tool_type
# tool type

from pydantic import BaseModel
from typing import Callable, Type, Dict, Any, Union


# completion tool type
CompletionToolType = Union[str, Callable, Type[BaseModel], Dict[str, Any]]