# tool type

from pydantic import BaseModel
from typing import Callable, Type, Dict, Any, Union


# completion tool type
CompletionsToolType = Union[str, Callable, Type[BaseModel], Dict[str, Any]]