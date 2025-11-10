"""zyx.core.processing.schemas.pydantic"""

from __future__ import annotations

import functools
import inspect
import json
from typing import Any, Callable, Type, TypeVar

from docstring_parser import parse
from instructor.dsl.simple_type import ModelAdapter, is_simple_type
from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
)

from ...._internal._exceptions import (
    ProcessingError,
    SchemaValidationError,
)
from .semantics import (
    to_semantic_description,
    to_semantic_key,
    to_semantic_title,
)

__all__ = [
    "snake_to_pascal",
    "function_to_pydantic_model",
    "to_pydantic_model",
    "validate_with_pydantic_model",
]


T = TypeVar("T")


InputT = TypeVar("InputT")
"""Alias representation for any kind of data used as the input to the validation function."""


ValidatorT = TypeVar("ValidatorT")
"""Alias representation for the object or type to use as the validation type."""


def snake_to_pascal(name: str) -> str:
    return "".join(word.capitalize() for word in name.split("_"))


def function_to_pydantic_model(
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    skip_first_param: bool = False,
) -> Type[BaseModel]:
    """
    Convert a function to a Pydantic model of it's parameters.

    Parameters
    ----------
        func: The function to convert
        name: Optional name for the model (defaults to PascalCase of func.__name__)
        skip_first_param: Skip first param (for methods with self/cls)

    Returns
    -------
        A Pydantic model with fields matching the function parameters
    """
    sig = inspect.signature(func)
    docstring = parse(func.__doc__ or "")

    # Build field definitions
    fields: dict[str, Any] = {}
    params = list(sig.parameters.items())

    if skip_first_param and params:
        params = params[1:]

    for param_name, param in params:
        # Get type annotation
        annotation = (
            param.annotation
            if param.annotation != inspect.Parameter.empty
            else Any
        )

        # Get description from docstring
        param_doc = next(
            (
                p.description
                for p in docstring.params
                if p.arg_name == param_name
            ),
            None,
        )

        # Build field with description
        if param.default != inspect.Parameter.empty:
            fields[param_name] = (
                annotation,
                Field(default=param.default, description=param_doc),
            )
        else:
            fields[param_name] = (
                annotation,
                Field(description=param_doc) if param_doc else ...,
            )

    model_name = name or snake_to_pascal(func.__name__)

    if description:
        fields["__doc__"] = (str, description)

    return create_model(model_name, **fields)  # type: ignore


def to_pydantic_model(
    source: T | Type[T],
    name: str | None = None,
    description: str | None = None,
    key: str | None = None,
) -> Type[BaseModel]:
    """
    Create a Pydantic model from any Python type, value, or function.

    This function is cached for immutable sources (types, functions).
    Non-hashable sources like dicts bypass the cache.

    Parameters
    ----------
        source: Can be a type, value, dict, function, etc.
        name: Optional name for the generated model
        description: Optional description for the generated model
        key: For simple types (int, str, etc.), override instructor's default "content" field name

    Returns
    -------
        A Pydantic BaseModel class

    Examples
    --------
        ```python
        >>> # Type
        >>> Model = to_pydantic_model(int)
        >>> Model(content=42)

        >>> # Type with custom key
        >>> Model = to_pydantic_model(int, key="value")
        >>> Model(value=42)

        >>> # Dict
        >>> Model = to_pydantic_model({"name": str, "age": int})
        >>> Model(name="John", age=30)

        >>> # Function
        >>> def search(query: str) -> list[str]: pass
        >>> Model = to_pydantic_model(search)
        >>> Model(query="test")
        ```
    """
    # Handle dict sources (not hashable, so can't be cached)
    if isinstance(source, dict):
        # Dictionary can be either:
        # 1. Field definitions (name -> type)
        # 2. Example data (name -> value) - infer types
        model_name = name or "Model"
        field_definitions = {}

        for field_name, value in source.items():
            # Check if value is a type or an actual value
            if isinstance(value, type):
                # It's a type definition
                field_definitions[field_name] = (value, ...)
            else:
                # It's an actual value, infer the type
                field_type = type(value)
                # Use the value as the default
                field_definitions[field_name] = (
                    field_type,
                    Field(default=value),
                )

        model = create_model(model_name, **field_definitions)  # type: ignore

        if description:
            model.__doc__ = description

        return model

    # Use cached internal implementation for hashable types
    @functools.lru_cache(maxsize=1000)
    def _cached(src: Any, params: tuple) -> Type[BaseModel]:
        """Internal cached implementation. params is (name, key)."""
        _name, _key = params

        # If already a Pydantic model, return as-is
        if isinstance(src, type) and issubclass(src, BaseModel):
            return src

        # If it's a function, convert it to a parameter model
        if inspect.isfunction(src) or inspect.ismethod(src):
            return function_to_pydantic_model(src, name=_name)

        # Use instructor's ModelAdapter/prepare_response_model
        try:
            # Check for simple types first (includes Literal, primitives, etc.)
            if is_simple_type(src):
                model = ModelAdapter[src]

                if not _name:
                    _name = to_semantic_title(src)

                if not _key:
                    _key = to_semantic_key(src)

                # remove instructor's default docstring and use semantic description
                # instructor uses 'correctly formatted ... response' which is not helpful
                model.__doc__ = to_semantic_description(src)

                if _key:
                    field_info = model.model_fields["content"]
                    model = create_model(
                        _name,
                        __doc__=model.__doc__,
                        **{_key: (field_info.annotation, field_info)},
                    )
                    return model
            else:
                # For complex types, use prepare_response_model
                from instructor.processing.response import (
                    prepare_response_model,
                )

                model = prepare_response_model(src)

            if _name and hasattr(model, "__name__"):
                model.__name__ = _name

            return model
        except Exception as e:
            raise ProcessingError(
                f"Cannot create Pydantic model from {src}"
            ) from e

    # Call cached function with tuple of parameters
    model = _cached(source, (name, key))

    # Apply description if provided (can't cache this part easily)
    if description and hasattr(model, "__doc__"):
        model.__doc__ = description

    return model


def validate_with_pydantic_model(
    data: InputT, source: ValidatorT, *, strict: bool = False
) -> ValidatorT:
    """
    Universal validation function that intelligently handles ANY combination of data and source types.

    This is a more powerful and flexible version of validate_with_schema that:
    - Auto-parses JSON strings to dicts
    - Handles model validation with both dict and **kwargs
    - Supports simple type validation
    - Uses cached models for performance
    - Intelligently unwraps single-field models

    Parameters
    ----------
    data : Any
        The data to validate. Can be:
        - Primitive values (int, str, bool, etc.)
        - Dicts (will be validated as model fields)
        - JSON strings (will be parsed then validated)
        - Lists/arrays
        - Nested structures
    source : Any
        The type/schema to validate against. Can be:
        - Python types (int, str, list[str], etc.)
        - Pydantic models
        - Functions (validates against parameter schema)
        - Complex types (dict[str, int], etc.)
    strict : bool, optional
        If True, use Pydantic's strict validation mode

    Returns
    -------
        The validated and parsed data, typed according to the source

    Raises:
    -------
        ValidationError: If the data doesn't match the schema
        ValueError: If JSON string is malformed

    Examples
    --------
        ```python
        >>> # Simple types
        >>> validate_with_model(42, int)
        42
        >>> validate_with_model("42", int)
        42

        >>> # JSON string to model
        >>> validate_with_model('{"name": "John", "age": 30}', User)
        User(name='John', age=30)

        >>> # Dict to model
        >>> validate_with_model({"name": "John", "age": 30}, User)
        User(name='John', age=30)

        >>> # String to complex type
        >>> validate_with_model("[1, 2, 3]", list[int])
        [1, 2, 3]

        >>> # Dict with extra validation
        >>> validate_with_model({"query": "search", "limit": 10}, search_function)
        SearchParams(query='search', limit=10)
        ```
    """
    # Step 1: Pre-process the data
    # If data is a JSON string, try to parse it
    processed_data = data
    if isinstance(data, str):
        # Try to parse as JSON first (for dicts, lists, etc.)
        # But only if it looks like JSON (starts with {, [, or quotes)
        stripped = data.strip()
        if stripped and stripped[0] in ("{", "[", '"'):
            try:
                processed_data = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON, keep as string
                # Will be validated as string below
                pass

    # Handle dict sources - can't be cached
    if isinstance(source, dict):
        model = to_pydantic_model(source)

        try:
            # Data is already an instance of the model
            if isinstance(processed_data, model):
                return processed_data

            # Data is a dict - validate as keyword arguments
            if isinstance(processed_data, dict):
                if strict:
                    return model.model_validate(
                        processed_data, strict=True
                    )
                else:
                    return model(**processed_data)

            # Try model_validate for other data types
            if strict:
                return model.model_validate(processed_data, strict=True)
            else:
                return model.model_validate(processed_data)

        except ValidationError as e:
            raise SchemaValidationError(
                f"Validation failed for {model.__name__}",
                errors=e.errors(),
            ) from e

    # Use cached internal implementation for hashable sources
    @functools.lru_cache(maxsize=1000)
    def _get_validator(src: Any, params: tuple):
        """Get model and field info for validation. params is (strict,)."""
        # Skip this for simple types - use TypeAdapter directly for speed
        if is_simple_type(src):
            return ("simple", TypeAdapter(src), None)

        # For everything else, use the cached model creation
        model = to_pydantic_model(src)
        fields = list(model.model_fields.keys())

        return ("model", model, fields)

    validator_type, validator, fields = _get_validator(source, (strict,))

    # Step 2: Handle simple types with TypeAdapter
    if validator_type == "simple":
        context = {"strict": strict} if strict else None
        return validator.validate_python(processed_data, context=context)

    # Step 3: Handle model validation
    model = validator
    try:
        # Strategy 1: Data is already an instance of the model
        if isinstance(processed_data, model):
            return processed_data

        # Strategy 2: Data is a dict - validate as keyword arguments
        if isinstance(processed_data, dict):
            if strict:
                return model.model_validate(processed_data, strict=True)
            else:
                return model(**processed_data)

        # Strategy 3: Data is a primitive or simple value
        # Check if this is a single-field model (wrapper around a primitive)
        if len(fields) == 1:
            # Single field model - likely wrapping a simple type
            field_name = fields[0]

            # Validate with the single field
            if strict:
                validated = model.model_validate(
                    {field_name: processed_data}, strict=True
                )
            else:
                validated = model(**{field_name: processed_data})

            # Extract and return the wrapped value
            return getattr(validated, field_name)

        # Strategy 4: Multi-field model but data is not a dict
        # Try model_validate which is more flexible
        if strict:
            return model.model_validate(processed_data, strict=True)
        else:
            # Try to coerce into the model
            try:
                return model.model_validate(processed_data)
            except (ValidationError, ValueError):
                # Last resort: wrap in dict if it's a single value
                raise TypeError(
                    f"Cannot validate {type(processed_data).__name__} against model {model.__name__} "
                    f"with fields {fields}. Expected a dict or compatible structure."
                )

    except ValidationError as e:
        raise SchemaValidationError(
            f"Validation failed for {model.__name__}", errors=e.errors()
        ) from e
