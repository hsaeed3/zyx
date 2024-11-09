# completion context type

from pydantic import BaseModel
from typing import Optional, Union, Type, Dict, List


# completion context type
CompletionsContext = Optional[Union[Type[BaseModel], BaseModel, Dict, List, str]]
