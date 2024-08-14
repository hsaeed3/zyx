from typing import Any, Dict, List, Type, Union
from ..core.ext import BaseModel


def convert_to_openai_tool(
    function: Union[Dict[str, Any], Type[BaseModel], callable],
) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI tool."""
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


def convert_python_function_to_openai_function(function: callable) -> Dict[str, Any]:
    """Convert a Python function to an OpenAI function-calling API compatible dict."""
    func_name = function.__name__
    annotations = function.__annotations__
    docstring = function.__doc__ or ""
    description, arg_descriptions = parse_docstring(docstring, list(annotations))
    parameters = {"type": "object", "properties": {}, "required": []}
    for arg, arg_type in annotations.items():
        if arg != "return":
            parameters["properties"][arg] = {
                "type": get_openai_type(arg_type),
                "description": arg_descriptions.get(arg, ""),
            }
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
        if line.lower().startswith("args:"):
            parsing_args = True
        elif parsing_args and ":" in line:
            arg, desc = line.split(":", 1)
            arg_descriptions[arg.strip()] = desc.strip()
        elif not parsing_args:
            description_lines.append(line)
    description = " ".join(description_lines).strip()
    return description, arg_descriptions


def get_openai_type(python_type: Type) -> str:
    """Convert Python type to OpenAI type."""
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return type_map.get(python_type, "string")
