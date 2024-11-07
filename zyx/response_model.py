# zyx.response_model
# converter methods for response_model

from pydantic import BaseModel, create_model
from typing import Type, Union, List
from .lib.exception import ZYXException


# HACK
# HACK
def make_nice_with_instructor(messages: list) -> list[dict[str, str]]:
    formatted_messages = []
    tool_output_map = {}

    # First pass: Collect tool outputs
    for message in messages:
        if message['role'] == 'tool':
            tool_call_id = message.get('tool_call_id')
            tool_output = message.get('content', 'unknown output')
            if tool_call_id:
                tool_output_map[tool_call_id] = tool_output

    # Second pass: Format messages
    for message in messages:
        if message['role'] == 'assistant' and message.get('tool_calls'):
            for tool_call in message['tool_calls']:
                tool_name = tool_call.get('function', {}).get('name', 'unknown tool')
                tool_args = tool_call.get('function', {}).get('arguments', 'unknown arguments')
                tool_call_id = tool_call.get('id')
                tool_output = tool_output_map.get(tool_call_id, 'unknown output')
                formatted_messages.append({
                    "role": "assistant",
                    "content": f"I executed the {tool_name} tool with the arguments {tool_args} and got the following output: {tool_output}."
                })
        elif message['role'] != 'tool':  # Skip messages with the role 'tool'
            formatted_messages.append(message)
    
    if formatted_messages and formatted_messages[-1]['role'] != 'user':
        formatted_messages.append({
            "role": "user",
            "content": "Proceed."
        })
    
    return formatted_messages


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


def create_patch_model(response_model_instance: BaseModel) -> Type[BaseModel]:
    """
    Creates a pydantic model with the same name as the input response_model_instance,
    but only includes fields that are currently empty.

    Args:
        response_model_instance (BaseModel): The input pydantic model instance.

    Returns:
        Type[BaseModel]: The created pydantic model with only empty fields.
    """
    empty_fields = {
        field_name: (field.type_, ...)
        for field_name, field in response_model_instance.model_fields.items()
        if getattr(response_model_instance, field_name) in (None, '', [], {}, set())
    }
    
    model_name = response_model_instance.__class__.__name__
    return create_model(model_name, **empty_fields)


def merge_models(x: BaseModel, y: BaseModel) -> BaseModel:
    """
    Merges two pydantic models with the same fields, filling empty fields in x with values from y.

    Args:
        x (BaseModel): The first pydantic model.
        y (BaseModel): The second pydantic model.

    Returns:
        BaseModel: The merged pydantic model with empty fields filled.
    """
    try:
        if x.__class__ != y.__class__:
            raise ValueError("Both models must be instances of the same class.")

        merged_data = x.model_dump()
        for field_name, value in y.model_dump().items():
            if merged_data[field_name] in (None, '', [], {}, set()):
                merged_data[field_name] = value

        return x.__class__(**merged_data)
    except Exception as e:
        raise ZYXException(f"Failed to merge models: {e}")


def create_dynamic_response_model(response_model: Union[str, List[Union[str, List[str]]]]) -> Type[BaseModel]:
    """
    Creates a pydantic model with fields based on the input string or list of strings.
    Supports nested lists to create nested pydantic models.

    Args:
        response_model (Union[str, List[Union[str, List[str]]]]): A string or a list of strings or nested lists representing the field names.

    Returns:
        Type[BaseModel]: The created pydantic model.
    """
    try:
        if isinstance(response_model, str):
                return create_model('Response', **{response_model: (str, ...)})
        elif isinstance(response_model, list):
            fields = {}
            for name in response_model:
                if isinstance(name, str):
                    fields[name] = (str, ...)
                elif isinstance(name, list):
                    nested_model = create_dynamic_response_model(name)
                    fields[f"nested_{response_model.index(name)}"] = (nested_model, ...)
                else:
                    raise ValueError("List elements must be strings or lists of strings.")
            return create_model('Response', **fields)
        else:
            raise ValueError("Input must be a string or a list of strings or nested lists.")
    except Exception as e:
        raise ZYXException(f"Failed to create dynamic response model: {e}")


# tests
if __name__ == "__main__":

    # Example 1: Merging two models with empty fields filled
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int = None

    user1 = User(name="Alice")
    user2 = User(name="Bob", age=30)
    merged_user = merge_models(user1, user2)
    print(merged_user.dict())

    # Example 2: Creating a simple dynamic response model with a single field
    model = create_dynamic_response_model("name")
    print(model.schema_json(indent=2))

    # Example 3: Creating a dynamic response model with multiple fields
    model = create_dynamic_response_model(["name", "age", "email"])
    print(model.schema_json(indent=2))

    # Example 4: Creating a nested dynamic response model
    model = create_dynamic_response_model(["name", ["address", "city", "zipcode"]])
    print(model.schema_json(indent=2))