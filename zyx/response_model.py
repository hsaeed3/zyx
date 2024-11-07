# zyx.response_model
# converter methods for response_model

from pydantic import BaseModel, create_model
from typing import Type, Union, List
from .lib.exception import ZYXException


# converter
def handle_response_model(response_model: Union[Type[BaseModel], Type, str, List[str]]) -> Type[BaseModel]:
    """
    Takes in a pydantic model (type) or a type, or a string or list of strings; 
    if it's already a pydantic model, returns the model; otherwise uses our functions to build, then returns.

    Args:
        response_model (Union[Type[BaseModel], Type, str, List[str]]): The input type to process.

    Returns:
        Type[BaseModel]: The pydantic model.
    """
    try:
        if isinstance(response_model, type) and issubclass(response_model, BaseModel):
            return response_model
        elif isinstance(response_model, type):
            return create_response_model(response_model)
        elif isinstance(response_model, (str, list)):
            return create_dynamic_response_model(response_model)
        else:
            raise ValueError("Input must be a pydantic model, a type, a string, or a list of strings.")
    except Exception as e:
        raise ZYXException(f"Failed to get or create response model: {e}")
    

def create_response_model(response_model: Type) -> Type[BaseModel]:
    """
    Creates a pydantic model with a single field 'response' of the given type.

    Args:
        response_model (Type): The type of the 'response' field.

    Returns:
        Type[BaseModel]: The created pydantic model.
    """
    return create_model('Response', response=(response_model, ...))


def create_dynamic_response_model(response_model: Union[str, List[str]]) -> Type[BaseModel]:
    """
    Creates a pydantic model with fields based on the input string or list of strings.

    Args:
        response_model (Union[str, List[str]]): A string or a list of strings representing the field names.

    Returns:
        Type[BaseModel]: The created pydantic model.
    """
    if isinstance(response_model, str):
        return create_model('Response', **{response_model: (str, ...)})
    elif isinstance(response_model, list):
        return create_model('Response', **{name: (str, ...) for name in response_model})
    else:
        raise ValueError("Input must be a string or a list of strings.")
