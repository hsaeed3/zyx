# zyx.types.completions.completion_context
# completion context type

from pydantic import BaseModel
from typing import Optional, Union, Type, Dict, List


# completion context type
CompletionContext = Optional[Union[Type[BaseModel], BaseModel, Dict, List, str]]
