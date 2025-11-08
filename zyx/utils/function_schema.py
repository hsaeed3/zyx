"""zyx.utils.function_schema

Enhanced OpenAI function schema generation with docstring parsing,
structured output support, and execution capabilities.
"""

from __future__ import annotations

import enum
import inspect
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, MISSING
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    Dict,
    Literal,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    TYPE_CHECKING,
)

from docstring_parser import parse
from pydantic import BaseModel, create_model

try:
    from types import UnionType
except ImportError:
    UnionType = Union  # type: ignore

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

__all__ = [
    "FunctionSchema",
    "get_function_schema",
]


ObjectJsonSchema: TypeAlias = dict[str, Any]
"""Type representing JSON schema of an object, e.g. where `"type": "object"`."""


T = TypeVar("T")


@dataclass
class FunctionSchema:
    """Represents a function schema with execution and rendering capabilities.

    This class encapsulates a function, its Pydantic model representation,
    and the JSON schema, providing methods to render OpenAI-compatible schemas
    and execute the function with parameters.
    """

    _function: Callable[..., Any] | None
    _model: Type[BaseModel]
    _schema: ObjectJsonSchema
    _takes_context: bool = False
    _function_name: str | None = None
    _cached_sig: inspect.Signature | None = None

    def function(self) -> Callable[..., Any] | Any:
        """Returns the underlying function, or a value getter for simple types.

        For simple types and models without __call__, returns a function that
        directly returns the value.
        """
        if self._function is not None:
            return self._function

        # For simple types/models without calls, create a direct value returner
        def _value_getter(value: Any) -> Any:
            """Returns the value directly for simple types."""
            return value

        return _value_getter

    def model(self) -> Type[BaseModel]:
        """Returns the Pydantic BaseModel representation of the schema."""
        return self._model

    @property
    def schema(self) -> ObjectJsonSchema:
        """Returns the base JSON schema representation."""
        return self._schema

    def render_openai_schema(
        self,
        name: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
    ) -> ChatCompletionToolParam:
        """Renders an OpenAI-compatible tool schema.

        Args:
            name: Optional name override for the function.
            description: Optional description override for the function.
            exclude: Optional set of parameter names to exclude from the schema.

        Returns:
            A ChatCompletionToolParam dictionary for OpenAI API calls.
        """
        # Determine function name early
        if name is None:
            name = self._function_name or (
                self._function.__name__
                if self._function is not None
                else self._model.__name__
            )

        # Determine function description early
        if description is None:
            if self._function is not None and self._function.__doc__:
                # Use docstring parser to extract short description and returns
                docstring = parse(self._function.__doc__)
                parts = []

                if docstring.short_description:
                    parts.append(docstring.short_description)

                if docstring.returns and docstring.returns.description:
                    parts.append(f"\n\n**Returns**: {docstring.returns.description}")

                description = (
                    " ".join(parts) if parts else self._function.__doc__.strip()
                )
            else:
                description = self._model.__doc__ or ""

        # Build the filtered schema efficiently
        if exclude:
            properties = self._schema.get("properties", {})
            required = self._schema.get("required", [])

            filtered_schema = {
                "type": "object",
                "properties": {k: v for k, v in properties.items() if k not in exclude},
                "required": [req for req in required if req not in exclude],
            }
        else:
            filtered_schema = self._schema

        # Build the ChatCompletionToolParam
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": filtered_schema,
            },
        }

    def execute(
        self,
        params: Dict[str, Any],
        use_function_defaults: bool = True,
        raise_on_error: bool = False,
    ) -> Any:
        """Executes the function with the provided parameters.

        Args:
            params: Dictionary of parameter names to values.
            use_function_defaults: If True, uses default values for missing parameters.
            raise_on_error: If False, returns an error dict instead of raising exceptions.

        Returns:
            The result of the function execution, or an error dict if raise_on_error is False.
        """
        func = self._function

        if func is None:
            # For simple types, just validate and return the value
            try:
                validated = self._model(**params)
                return (
                    validated.model_dump()
                    if hasattr(validated, "model_dump")
                    else params
                )
            except Exception as e:
                if raise_on_error:
                    raise
                return {"error": str(e), "type": type(e).__name__}

        try:
            # Get cached signature or create new one
            if self._cached_sig is None:
                self._cached_sig = inspect.signature(func)
            sig = self._cached_sig

            # Build arguments efficiently
            bound_args = {}
            param_items = list(sig.parameters.items())

            # Skip first parameter if takes_context
            if self._takes_context and param_items:
                param_items = param_items[1:]

            # Build arguments with defaults if needed
            for param_name, param in param_items:
                if param_name in params:
                    bound_args[param_name] = params[param_name]
                elif (
                    use_function_defaults
                    and param.default is not inspect.Parameter.empty
                ):
                    bound_args[param_name] = param.default
                elif param.default is inspect.Parameter.empty:
                    # Required parameter missing
                    if raise_on_error:
                        raise ValueError(f"Missing required parameter: {param_name}")
                    return {
                        "error": f"Missing required parameter: {param_name}",
                        "type": "ValueError",
                    }

            # Execute the function
            return func(**bound_args)

        except Exception as e:
            if raise_on_error:
                raise
            return {"error": str(e), "type": type(e).__name__}


def get_function_schema(
    source: Callable[..., Any] | Type[BaseModel] | Type[Any] | dict[str, Any],
    takes_context: bool = False,
) -> FunctionSchema:
    """Creates a FunctionSchema from various input types.

    Supports:
    - Regular functions with type annotations
    - Pydantic BaseModel classes (with or without __call__ method)
    - TypedDict classes
    - Regular dictionaries with values
    - Simple types (via instructor's ModelAdapter)
    - Dataclasses

    Args:
        source: The source to convert into a FunctionSchema.
        takes_context: If True, skips the first parameter (context parameter) in the schema.

    Returns:
        A FunctionSchema instance with function, model, and schema representations.
    """
    # Handle regular functions (most common case first for speed)
    if inspect.isfunction(source) or inspect.ismethod(source):
        return _create_function_schema_from_function(source, takes_context)

    # Handle dataclass (check before BaseModel as it's more specific)
    if is_dataclass(source):
        return _create_function_schema_from_dataclass(source)

    # Handle Pydantic BaseModel with __call__ method
    if isinstance(source, type):
        try:
            if issubclass(source, BaseModel):
                # Check if __call__ is defined on this class (not inherited from BaseModel)
                if "__call__" in source.__dict__:
                    # Has __call__, treat as callable model
                    return _create_function_schema_from_callable_model(
                        source, takes_context
                    )
                else:
                    # Regular BaseModel, create schema from fields
                    return _create_function_schema_from_basemodel(source)
        except TypeError:
            # Not a class, continue
            pass

    # Handle TypedDict
    if hasattr(source, "__annotations__") and "__required_keys__" in dir(source):
        return _create_function_schema_from_typeddict(source)

    # Handle regular dict
    if isinstance(source, dict):
        return _create_function_schema_from_dict(source)

    # Handle simple types via instructor's ModelAdapter
    try:
        from instructor.processing.response import prepare_response_model

        model = prepare_response_model(source)
        schema = model.model_json_schema()

        # Convert model schema to function parameter schema
        param_schema = _convert_model_schema_to_param_schema(schema)

        return FunctionSchema(
            _function=None,
            _model=model,
            _schema=param_schema,
            _takes_context=False,
        )
    except Exception as e:
        raise ValueError(f"Unsupported source type: {type(source)}") from e


def _create_function_schema_from_function(
    func: Callable[..., Any],
    takes_context: bool = False,
) -> FunctionSchema:
    """Creates a FunctionSchema from a regular function."""
    # Get the enhanced schema using docstring parsing
    schema = _get_enhanced_function_schema(func, takes_context)

    # Create Pydantic model from function
    model = _convert_function_to_pydantic_basemodel(func, takes_context)

    return FunctionSchema(
        _function=func,
        _model=model,
        _schema=schema,
        _takes_context=takes_context,
        _function_name=func.__name__,
        _cached_sig=None,  # Will be cached on first execute()
    )


def _create_function_schema_from_callable_model(
    model_class: Type[BaseModel],
    takes_context: bool = False,
) -> FunctionSchema:
    """Creates a FunctionSchema from a Pydantic model with __call__ method."""
    call_method = getattr(model_class, "__call__")

    # Get schema from __call__ method, skipping 'self' parameter
    # We need to skip 'self' (first param), so we set takes_context=True
    # to skip it, then restore the original takes_context value
    schema = _get_enhanced_function_schema(call_method, takes_context=True)

    # Create wrapper function that instantiates and calls
    def _callable_wrapper(**kwargs):
        instance = model_class()
        return instance(**kwargs)

    # Create model from __call__ method, skipping 'self' parameter
    model = _convert_function_to_pydantic_basemodel(call_method, takes_context=True)

    return FunctionSchema(
        _function=_callable_wrapper,
        _model=model,
        _schema=schema,
        _takes_context=takes_context,
        _function_name=model_class.__name__,
        _cached_sig=None,
    )


def _create_function_schema_from_basemodel(
    model_class: Type[BaseModel],
) -> FunctionSchema:
    """Creates a FunctionSchema from a regular Pydantic BaseModel."""
    # Get schema from model
    model_schema = model_class.model_json_schema()
    param_schema = _convert_model_schema_to_param_schema(model_schema)

    # Create a simple getter function
    def _model_getter(**kwargs) -> dict[str, Any]:
        instance = model_class(**kwargs)
        return instance.model_dump()

    return FunctionSchema(
        _function=_model_getter,
        _model=model_class,
        _schema=param_schema,
        _takes_context=False,
        _function_name=model_class.__name__,
        _cached_sig=None,
    )


def _create_function_schema_from_typeddict(
    typeddict_class: Type[Any],
) -> FunctionSchema:
    """Creates a FunctionSchema from a TypedDict."""
    annotations = typeddict_class.__annotations__
    required_keys = getattr(typeddict_class, "__required_keys__", set())

    # Build schema
    properties = {}
    for key, type_hint in annotations.items():
        type_value = _guess_type(type_hint)
        properties[key] = {
            "description": f"The {key} parameter",
        }
        if type_value is not None:
            properties[key]["type"] = type_value

    schema = {
        "type": "object",
        "properties": properties,
        "required": list(required_keys),
    }

    # Create Pydantic model
    field_definitions = {
        k: (v, ... if k in required_keys else None) for k, v in annotations.items()
    }
    model = create_model(
        typeddict_class.__name__,
        **field_definitions,
    )

    def _typeddict_getter(**kwargs) -> dict[str, Any]:
        return kwargs

    return FunctionSchema(
        _function=_typeddict_getter,
        _model=model,
        _schema=schema,
        _takes_context=False,
        _function_name=typeddict_class.__name__,
        _cached_sig=None,
    )


def _create_function_schema_from_dict(
    dict_source: dict[str, Any],
) -> FunctionSchema:
    """Creates a FunctionSchema from a regular dictionary."""
    # Build schema from dict structure
    properties = {}
    field_definitions = {}

    for key, value in dict_source.items():
        value_type = type(value)
        type_value = _guess_type(value_type)

        prop = {
            "description": f"The {key} parameter",
            "default": value,
        }
        if type_value is not None:
            prop["type"] = type_value

        properties[key] = prop
        field_definitions[key] = (value_type, value)

    schema = {
        "type": "object",
        "properties": properties,
        "required": list(dict_source.keys()),
    }

    # Create Pydantic model
    model = create_model("DictModel", **field_definitions)

    def _dict_getter(**kwargs) -> dict[str, Any]:
        return kwargs

    return FunctionSchema(
        _function=_dict_getter,
        _model=model,
        _schema=schema,
        _takes_context=False,
        _function_name="DictModel",
        _cached_sig=None,
    )


def _create_function_schema_from_dataclass(
    dataclass_type: Type[Any],
) -> FunctionSchema:
    """Creates a FunctionSchema from a dataclass."""
    dc_fields = fields(dataclass_type)

    # Build schema
    properties = {}
    required = []
    field_definitions = {}

    for field in dc_fields:
        field_type = field.type
        type_value = _guess_type(field_type)

        prop_schema = {
            "description": f"The {field.name} parameter",
        }
        if type_value is not None:
            prop_schema["type"] = type_value

        # Handle default values
        if field.default is not MISSING:
            prop_schema["default"] = field.default
            field_definitions[field.name] = (field_type, field.default)
        elif field.default_factory is not MISSING:
            default_val = field.default_factory()
            prop_schema["default"] = default_val
            field_definitions[field.name] = (field_type, default_val)
        else:
            required.append(field.name)
            field_definitions[field.name] = (field_type, ...)

        properties[field.name] = prop_schema

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    # Create Pydantic model
    model = create_model(
        dataclass_type.__name__,
        __doc__=dataclass_type.__doc__,
        **field_definitions,
    )

    def _dataclass_getter(**kwargs):
        return dataclass_type(**kwargs)

    return FunctionSchema(
        _function=_dataclass_getter,
        _model=model,
        _schema=schema,
        _takes_context=False,
        _function_name=dataclass_type.__name__,
        _cached_sig=None,
    )


@lru_cache(maxsize=1000)
def _get_enhanced_function_schema(
    func: Callable[..., Any],
    takes_context: bool = False,
) -> ObjectJsonSchema:
    """Get function parameter schema in JSON Schema format with enhanced docstring parsing.

    This method is heavily inspired by the logic within the `function_schema` package.

    Extracts function signature and docstring information to create a JSON schema
    for the function's parameters, suitable for OpenAI function calling.

    Args:
        func: The function to get the schema for.
        takes_context: If True, skips the first parameter (context parameter) in the schema.

    Returns:
        A JSON Schema object representing the function's parameters.
    """
    # Parse the docstring to extract descriptions
    docstring = parse(func.__doc__ or "")

    # Extract parameter descriptions from docstring
    param_descriptions: dict[str, str] = {}
    for param in docstring.params:
        if param.arg_name and param.description:
            # Strip whitespace from parameter names to handle various docstring formats
            param_descriptions[param.arg_name.strip()] = param.description

    # Get function signature
    sig = inspect.signature(func)
    params = sig.parameters

    # Skip first parameter if takes_context is True
    if takes_context:
        param_items = list(params.items())
        if param_items:
            param_items = param_items[1:]  # Remove first parameter
        params_to_process = dict(param_items)
    else:
        params_to_process = params

    # Initialize schema
    schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # Get type hints
    type_hints = get_type_hints(func, include_extras=True)

    # Process each parameter
    for name, param in params_to_process.items():
        type_hint = type_hints.get(name)
        if type_hint is not None:
            param_args = get_args(type_hint)
            is_annotated = get_origin(type_hint) is Annotated
        else:
            param_args = []
            is_annotated = False

        enum_ = None
        default_value = inspect._empty

        if is_annotated:
            # First arg is the actual type
            (T, *_) = param_args
            # Check for enum in annotations
            enum_ = next(
                (
                    [e.name for e in arg]
                    for arg in param_args
                    if isinstance(arg, type) and issubclass(arg, enum.Enum)
                ),
                # Use typing.Literal as enum if no enum found
                get_origin(T) is Literal and get_args(T) or None,
            )
        else:
            T = param.annotation
            if get_origin(T) is Literal:
                enum_ = get_args(T)

        # Get default value
        if param.default is not inspect._empty:
            default_value = param.default

        # Use docstring description if available, otherwise use generic description
        description = param_descriptions.get(name, f"The {name} parameter")

        # Build property schema
        type_value = _guess_type(T)
        if type_value is not None:
            schema["properties"][name] = {
                "type": type_value,
                "description": description,
            }
        else:
            # For Any or complex types
            schema["properties"][name] = {
                "description": description,
            }

        # Add enum if present
        if enum_ is not None:
            schema["properties"][name]["enum"] = [t for t in enum_ if t is not None]

        # Add default if present
        if default_value is not inspect._empty:
            schema["properties"][name]["default"] = default_value

        # Determine if required
        if (
            get_origin(T) is not Literal
            and not isinstance(None, T)
            and default_value is inspect._empty
        ):
            schema["required"].append(name)

        if get_origin(T) is Literal:
            if all(get_args(T)):
                schema["required"].append(name)

    # Deduplicate required fields
    schema["required"] = list(set(schema["required"]))

    return schema


def _convert_function_to_pydantic_basemodel(
    func: Callable[..., Any],
    takes_context: bool = False,
) -> Type[BaseModel]:
    """Converts a function's parameters to a Pydantic BaseModel.

    Args:
        func: A callable function.
        takes_context: If True, skips the first parameter.

    Returns:
        A Pydantic BaseModel class with fields corresponding to the function's parameters.
    """
    sig = inspect.signature(func)
    field_definitions = {}

    param_items = list(sig.parameters.items())
    if takes_context and param_items:
        param_items = param_items[1:]

    for param_name, param in param_items:
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{param_name}' is missing a type annotation.")

        param_type = param.annotation

        # Handle default values
        if param.default is not inspect.Parameter.empty:
            field_definitions[param_name] = (param_type, param.default)
        else:
            field_definitions[param_name] = (param_type, ...)

    model_name = func.__name__

    return create_model(
        model_name,
        __doc__=func.__doc__,
        **field_definitions,
    )


def _convert_model_schema_to_param_schema(
    model_schema: dict[str, Any],
) -> ObjectJsonSchema:
    """Converts a Pydantic model JSON schema to a function parameter schema.

    Args:
        model_schema: The model's JSON schema from model_json_schema().

    Returns:
        A parameter schema suitable for OpenAI function calling.
    """
    return {
        "type": "object",
        "properties": model_schema.get("properties", {}),
        "required": model_schema.get("required", []),
    }


@lru_cache(maxsize=500)
def _guess_type(T: type) -> str | list[str] | dict[str, Any] | None:
    """Guess the JSON schema type for a given Python type.

    Args:
        T: The Python type to convert.

    Returns:
        A JSON schema type string, list of type strings, dict for Any, or None for NoneType.
    """
    # Special case for Any
    if T is Any:
        return {}

    origin = get_origin(T)

    # Handle Annotated types
    if origin is Annotated:
        return _guess_type(get_args(T)[0])

    # Handle Union types
    if origin in [Union, UnionType]:
        union_types = [t for t in get_args(T) if t is not type(None)]
        # Avoid double calls to _guess_type
        _types = []
        seen = set()
        for union_type in union_types:
            type_val = _guess_type(union_type)
            if type_val is not None and type_val not in seen:
                _types.append(type_val)
                seen.add(type_val)

        if len(_types) == 1:
            return _types[0]
        return _types if _types else None

    # Handle Literal types
    if origin is Literal:
        type_args = Union[tuple(type(arg) for arg in get_args(T))]
        return _guess_type(type_args)

    # Handle container types
    if origin is list or origin is tuple:
        return "array"
    elif origin is dict:
        return "object"

    # Handle non-type objects
    if not isinstance(T, type):
        return None

    # Handle NoneType
    if T.__name__ == "NoneType":
        return None

    # Handle basic types (use try-except for faster checking)
    try:
        # Check bool before int (bool is subclass of int)
        if issubclass(T, bool):
            return "boolean"
        if issubclass(T, str):
            return "string"
        if issubclass(T, (float, int)):
            return "number"
    except TypeError:
        # Not a class
        pass

    # Handle by name for built-in types
    if T.__name__ == "list":
        return "array"
    if T.__name__ == "dict":
        return "object"

    return None
