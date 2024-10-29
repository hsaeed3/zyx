from typing import Any, Dict, List, Type, Union
from pydantic import BaseModel


def convert_to_openai_tool(
    function: Union[Dict[str, Any], Type[BaseModel], callable],
) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI tool.

    Parameters:
        function: A function, class, or dictionary representing the function.

    Returns:
        A dictionary representing the OpenAI tool / function.
    """
    tool = {}

    if isinstance(function, dict) and all(
        k in function for k in ("name", "description", "parameters")
    ):
        tool = function
    elif isinstance(function, dict) and all(
        k in function for k in ("title", "description", "properties")
    ):
        function = function.copy()
        tool = {
            "name": function.pop("title"),
            "description": function.pop("description"),
            "parameters": function,
        }
    elif isinstance(function, type) and issubclass(function, BaseModel):
        tool = convert_pydantic_to_openai_function(function)
    elif callable(function):
        tool = convert_python_function_to_openai_function(function)
    else:
        raise ValueError("Unsupported function type")

    return {"type": "function", "function": tool}


def convert_pydantic_to_openai_function(
    model: Type[BaseModel], name: str = None, description: str = None
) -> Dict[str, Any]:
    """Converts a Pydantic model to a function description for the OpenAI API."""
    schema = model.model_json_schema()
    schema.pop("definitions", None)
    title = schema.pop("title", "")
    default_description = schema.pop("description", "")
    return {
        "name": name or title,
        "description": description or default_description,
        "parameters": schema,
    }


from typing import Any, Dict, List, Type, Union, Optional
from pydantic import BaseModel


def convert_python_function_to_openai_function(function: callable) -> Dict[str, Any]:
    """Convert a Python function to an OpenAI function-calling API compatible dict."""
    func_name = function.__name__
    annotations = function.__annotations__
    docstring = function.__doc__ or ""

    description, arg_descriptions = parse_docstring(docstring, list(annotations))

    parameters = {"type": "object", "properties": {}, "required": []}
    for arg, arg_type in annotations.items():
        if arg != "return":
            openai_type = get_openai_type(arg_type)
            parameters["properties"][arg] = {
                "type": openai_type,
                "description": arg_descriptions.get(arg, ""),
            }
            # Check if the type is Optional
            if not is_optional_type(arg_type):
                parameters["required"].append(arg)

    return {"name": func_name, "description": description, "parameters": parameters}


def parse_docstring(docstring: str, args: List[str]) -> tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring."""
    description = ""
    arg_descriptions = {}
    lines = docstring.split("\n")
    description_lines = []
    parsing_args = False
    for line in lines:
        line = line.strip()
        if line.lower().startswith(("args:", "params:", ":params")):
            parsing_args = True
        elif parsing_args and ":" in line:
            arg, desc = line.split(":", 1)
            arg_descriptions[arg.strip()] = desc.strip()
        elif not parsing_args and not line.lower().startswith("returns:"):
            description_lines.append(line)
    description = " ".join(description_lines).strip()
    return description, arg_descriptions


def get_openai_type(python_type: Type) -> str:
    """Convert Python type to OpenAI type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Handle Optional and other generic types
    if hasattr(python_type, "__origin__"):
        origin = python_type.__origin__
        if origin is list:
            return "array"
        elif origin is dict:
            return "object"
        elif origin is Union and type(None) in python_type.__args__:
            # Handle Optional types (e.g., Optional[int] -> int)
            non_none_types = [
                arg for arg in python_type.__args__ if arg is not type(None)
            ]
            if non_none_types:
                return get_openai_type(non_none_types[0])

    return type_map.get(python_type, "string")


def is_optional_type(python_type: Type) -> bool:
    """Check if a type is Optional."""
    if hasattr(python_type, "__origin__") and python_type.__origin__ is Union:
        return type(None) in python_type.__args__
    return False
