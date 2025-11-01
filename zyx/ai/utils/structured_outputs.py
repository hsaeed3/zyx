"""zyx.ai.utils.structured_outputs

Central resource for most utility in relation to generating structured
outputs, along with direct resources from the instructor library.

`instructor` is one of the most important / fundamental dependency within
zyx, and is used at more levels than just generating structured outputs.
"""

from __future__ import annotations

from dataclasses import is_dataclass, fields, MISSING
from inspect import isfunction, signature, Parameter
from typing import Any, Callable, Type, TypeVar, Literal, TypeAliasType

from pydantic import BaseModel, create_model
from instructor import (
    Mode as InstructorMode,
    from_litellm,
    from_openai,
    AsyncInstructor,
)
from instructor.processing.response import (
    prepare_response_model as _prepare_response_model,
)

from ...core.exceptions import InstructorLibraryException

__all__ = [
    "StructuredOutputType",
    "InstructorModeName",
    "prepare_structured_output_model",
    # Instructor Re-Exports
    "InstructorMode",
    "from_openai",
    "from_litellm",
    "AsyncInstructor",
]


StructuredOutputType = TypeVar("StructuredOutputType")
"""Generic alias for the 'response_model' type parameter used by instructor
to define the schema or type of a structured output."""


InstructorModeName = TypeAliasType(
    "InstructorModeName",
    Literal[
        "function_call",  # deprecated as per instructor 1.11.3
        "parallel_tool_call",
        "tool_call",
        "tools_strict",
        "json_mode",
        "json_o1",
        "markdown_json_mode",
        "json_schema_mode",
        "responses_tools",
        "responses_tools_with_inbuilt_tools",
        "xai_json",
        "xai_tools",
        "anthropic_tools",
        "anthropic_reasoning_tools",
        "anthropic_json",
        "anthropic_parallel_tools",
        "mistral_tools",
        "mistral_structured_outputs",
        "vertexai_tools",
        "vertexai_json",
        "vertexai_parallel_tools",
        "gemini_json",
        "gemini_tools",
        "genai_tools",
        "genai_structured_outputs",
        "cohere_tools",
        "json_object",
        "cerebras_tools",
        "cerebras_json",
        "fireworks_tools",
        "fireworks_json",
        "writer_tools",
        "writer_json",
        "bedrock_tools",
        "bedrock_json",
        "perplexity_json",
        "openrouter_structured_outputs",
    ],
)
"""String alias for `instructor.mode.Mode`. There is no real
need for this to exist, but it is and we are."""


_RESPONSE_MODEL_CACHE = dict()
"""Cache for prepared response models to avoid redundant processing."""


def prepare_structured_output_model(
    t: Type[Any] | Any,
    name: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
) -> Type[BaseModel]:
    """Prepares a Pydantic BaseModel to be used as a structured output object based
    on a provided type or instance.
    This uses the default conversion logic from the `instructor` library for content such as
    simple types & explicitly handles functions and dataclasses.

    Caches prepared models to optimize performance on repeated calls with the same parameters.

    Args:
        t: The type or instance to convert into a Pydantic BaseModel.
        name: Optional name for the generated Pydantic model.
        description: Optional description for the generated Pydantic model.
        exclude: Optional set of field names to exclude from the model.

    Returns:
        A Pydantic BaseModel class representing the structured output.
    """

    # Custom cache to handle unhashable arguments
    def _get_cache_key(t, name, description, exclude):
        if hasattr(t, "__module__") and hasattr(t, "__name__"):
            t_key = (t.__module__, t.__name__)
        elif isinstance(t, dict):
            t_key = tuple(sorted((k, repr(v)) for k, v in t.items()))
        else:
            t_key = id(t)  # Fallback for other unhashable types
        return (t_key, name, description, frozenset(exclude or set()))

    key = _get_cache_key(t, name, description, exclude)
    if key in _RESPONSE_MODEL_CACHE:
        return _RESPONSE_MODEL_CACHE[key]

    if exclude is None:
        exclude = set()

    if isfunction(t):
        result = _convert_function_to_pydantic_basemodel(
            t, name=name, description=description, exclude=exclude
        )

    elif is_dataclass(t):
        result = _convert_dataclass_to_pydantic_basemodel(
            t, name=name, description=description, exclude=exclude
        )

    elif isinstance(t, type) and issubclass(t, BaseModel):
        # Remove excluded fields from BaseModel
        fields_dict = {
            k: (v.annotation, v.default if v.default is not None else ...)
            for k, v in t.__annotations__.items()
            if k not in exclude
        }
        result = create_model(
            name or t.__name__,
            __doc__=description if description is not None else t.__doc__,
            **fields_dict,
        )

    elif isinstance(t, dict):
        # Remove excluded keys from dict
        filtered = {k: v for k, v in t.items() if k not in exclude}
        result = create_model(
            name or "DictModel",
            __doc__=description,
            **{k: (type(v), ...) for k, v in filtered.items()},
        )

    elif hasattr(t, "__annotations__") and "__required_keys__" in dir(t):
        # TypedDict
        keys = [k for k in t.__annotations__.keys() if k not in exclude]
        result = create_model(
            name or t.__name__,
            __doc__=description if description is not None else t.__doc__,
            **{k: (t.__annotations__[k], ...) for k in keys},
        )

    else:
        try:
            model = _prepare_response_model(
                t,
            )
            if name is not None:
                model.__class__.__name__ = name
            if description is not None:
                model.__class__.__doc__ = description

            result = model
        except Exception as e:
            raise InstructorLibraryException(
                exception=e, message=f"Failed to prepare response model for type '{t}'."
            )

    _RESPONSE_MODEL_CACHE[key] = result
    return result


def _convert_function_to_pydantic_basemodel(
    t: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
) -> Type[BaseModel]:
    """Converts a function's parameters to a Pydantic BaseModel.

    Args:
        t: A callable function

    Returns:
        A Pydantic BaseModel class with fields corresponding to the function's parameters
    """
    if not callable(t):
        raise TypeError("Provided type is not a callable function.")

    sig = signature(t)
    field_definitions = {}
    exclude = exclude or set()
    for param_name, param in sig.parameters.items():
        if param_name in exclude:
            continue
        if param.annotation is Parameter.empty:
            raise TypeError(f"Parameter '{param_name}' is missing a type annotation.")
        param_type = param.annotation
        # Handle default values
        if param.default is not Parameter.empty:
            field_definitions[param_name] = (param_type, param.default)
        else:
            field_definitions[param_name] = (param_type, ...)
    model_name = t.__name__ if name is None else name

    return create_model(
        model_name,
        __doc__=description if description is not None else t.__doc__,
        **field_definitions,
    )


def _convert_dataclass_to_pydantic_basemodel(
    t: Type[Any] | Any,
    name: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
) -> Type[BaseModel]:
    """Converts a dataclass to a Pydantic BaseModel.

    Args:
        t: A dataclass type or instance

    Returns:
        A Pydantic BaseModel class with the same fields as the dataclass

    Raises:
        TypeError: If the provided type is not a dataclass
    """
    if not is_dataclass(t):
        raise TypeError("Provided type is not a dataclass.")

    # Get the dataclass type (handle both type and instance)
    dataclass_type = t if isinstance(t, type) else type(t)

    # Get dataclass fields
    dc_fields = fields(dataclass_type)
    exclude = exclude or set()
    # Build field definitions for Pydantic model
    field_definitions = {}
    for field in dc_fields:
        if field.name in exclude:
            continue
        field_type = field.type
        # Handle default values
        if field.default is not MISSING:
            field_definitions[field.name] = (field_type, field.default)
        elif field.default_factory is not MISSING:
            field_definitions[field.name] = (field_type, field.default_factory())
        else:
            field_definitions[field.name] = (field_type, ...)

    return create_model(
        dataclass_type.__name__ if name is None else name,
        __doc__=description if description is not None else dataclass_type.__doc__,
        **field_definitions,
    )
