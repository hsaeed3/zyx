# zyx.types.completions.completion_arguments
# completion arguments

from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Type, Union


# completion arguments
class CompletionArguments(BaseModel):

    """Base Completion Arguments Model"""

    class Config:
        arbitrary_types_allowed = True

    
    # messages
    messages : Union[str, Dict[str, Any], List[Dict[str, Any]], List[List[Dict[str, Any]]]]

    # model
    model : str


    # optional arguments
    # instructor mode
    mode : Optional[str] = None
    # structured output
    response_model: Optional[Union[BaseModel, Type[BaseModel], str, Dict[str, Any], Type[int], Type[float], Type[str], Type[bool], Type[list]]] = None
    # response format
    response_format : Optional[Union[BaseModel, Type[BaseModel], str, Dict[str, Any], Type[int], Type[float], Type[str], Type[bool], Type[list]]] = None


    # base completion arguments
    stream : Optional[bool] = False


    
