"""zyx.processing.outputs"""

from __future__ import annotations

import dataclasses
import sys
import types
from enum import Enum
from inspect import isclass
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    Literal,
    Iterable,
    Optional,
    Set,
    Type,
    Union,
    get_origin,
    get_args,
)

from pydantic import BaseModel, Field, create_model

__all__ = (
    "is_simple_type",
    "normalize_output_target",
    "partial_output_model",
    "selection_output_model",
    "sparse_output_model",
    "split_output_model",
)


_processing_models: Set[Type[BaseModel]] = set()


def is_simple_type(
    obj: type[BaseModel] | str | int | float | bool | Any,
) -> bool:
    """
    Helper method to determine if a given object is a simple type.

    NOTE: This code is taken directly from `instructor.dsl.simple_type`, you
    can view the original code here:
    [code](https://github.com/567-labs/instructor/blob/main/instructor/dsl/simple_type.py)
    """
    # Get the origin of the response model
    origin = get_origin(obj)

    # Handle special case for list[int | str], list[Union[int, str]] or similar type patterns
    # Identify a list type by checking for various origins it might have
    if origin in {Iterable, list}:
        # For list types, check the contents before deciding
        if origin is list:
            # Extract the inner types from the list
            args = get_args(obj)
            if args and len(args) == 1:
                inner_arg = args[0]
                # Special handling for Union types
                inner_origin = get_origin(inner_arg)

                # Explicit check for Union types - try different patterns across Python versions
                if (
                    inner_origin is Union
                    or inner_origin == Union
                    or str(inner_origin) == "Union"
                    or str(type(inner_arg)) == "<class '_UnionGenericAlias'>"
                ):
                    return True

                # Check for Python 3.10+ pipe syntax
                if hasattr(inner_arg, "__or__"):
                    return True

                # For simple list with basic types, also return True
                if inner_arg in {str, int, float, bool}:
                    return True

                # Check if inner type is a BaseModel - if so, not a simple type
                try:
                    if isclass(inner_arg) and issubclass(inner_arg, BaseModel):
                        return False
                except TypeError:
                    pass

            # If no args or unknown pattern, treat as simple list
            return len(args) == 0

        # Extract the inner types from the list for other iterable types
        args = get_args(obj)
        if args and len(args) == 1:
            inner_arg = args[0]
            # Special handling for Union types
            inner_origin = get_origin(inner_arg)

            # Explicit check for Union types - try different patterns across Python versions
            if (
                inner_origin is Union
                or inner_origin == Union
                or str(inner_origin) == "Union"
                or str(type(inner_arg)) == "<class '_UnionGenericAlias'>"
            ):
                return True

            # Check for Python 3.10+ pipe syntax
            if hasattr(inner_arg, "__or__"):
                return True

            # For simple list with basic types, also return True
            if inner_arg in {str, int, float, bool}:
                return True

        # For other iterable patterns, return False (e.g., streaming types)
        return False

    try:
        if obj in {str, int, float, bool}:
            return True
    except TypeError:
        # obj is unhashable (e.g. a dict/list value, not a type) — not simple
        return False

    # If the obj is a simple type like annotated
    # Build set of union types based on Python version
    union_types = {Union}
    if sys.version_info >= (3, 10):
        union_types.add(types.UnionType)

    if (
        origin
        in {
            Annotated,
            Literal,
            list,  # origin of List[T] is list
        }
        or origin in union_types
    ):
        return True

    if isclass(obj) and issubclass(obj, Enum):
        return True

    return False


@lru_cache(maxsize=128)
def _normalize_output_type(
    obj: Type[Any | BaseModel],
) -> Type[Any | BaseModel]:
    """
    Normalizes/validates a target type to ensure it is valid for use as
    the 'output_type' parameter when running a Pydantic AI agent.
    """
    if is_simple_type(obj):
        return obj

    if isinstance(obj, type) and issubclass(obj, BaseModel):
        return obj

    # --- dataclasses
    if dataclasses.is_dataclass(obj) and isinstance(obj, type):
        fields: dict[str, tuple[Any, Any]] = {}

        for dataclass_field in dataclasses.fields(obj):
            annotation = (
                dataclass_field.type
                if dataclass_field.type is not None
                else Any
            )

            if (
                dataclass_field.default is dataclasses.MISSING
                and dataclass_field.default_factory is dataclasses.MISSING
            ):
                default: Any = ...

            elif dataclass_field.default_factory is not dataclasses.MISSING:
                default = Field(
                    default_factory=dataclass_field.default_factory
                )
            else:
                default = Field(default=dataclass_field.default)

            fields[dataclass_field.name] = (annotation, default)

        model = create_model(obj.__name__ or "Response", **fields)  # type: ignore[call-overload]
        if obj.__doc__:
            model.__doc__ = obj.__doc__
        return model

    if isinstance(obj, type):
        return obj

    return type(obj)


def _normalize_output_value(value: Any | BaseModel) -> Type[Any | BaseModel]:
    """
    Normalizes/converts a value to a type representation that is compatible if
    used as the 'output_type' parameter when running a Pydantic AI agent.
    """
    # --- dictionaries (check before type conversion to avoid returning bare dict class)
    if isinstance(value, dict):
        fields: dict[str, tuple[Any, Any]] = {}
        for field_name, field_value in value.items():
            if isinstance(field_value, type) or is_simple_type(field_value):
                fields[field_name] = (field_value, ...)
            else:
                fields[field_name] = (
                    type(field_value),
                    Field(default=field_value),
                )
        return create_model("Response", **fields)  # type: ignore[call-overload]

    # --- dataclasses
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_output_type(type(value))

    try:
        return _normalize_output_type(type(value))

    except TypeError:
        return type(value)


def normalize_output_target(target: Type[Any] | Any) -> Type[Any | BaseModel]:
    """
    Normalizes/converts a target type or value to a type representation that is compatible if
    used as the 'output_type' parameter when running a Pydantic AI agent.

    Args:
        target : Type[Any] | Any
            The target type or value to normalize.

    Returns:
        Type[Any | BaseModel]
            The normalized target type.

    Raises:
        TypeError : If the target is not a valid type or value.
    """
    try:
        return _normalize_output_type(target)
    except TypeError:
        pass

    try:
        return _normalize_output_value(target)
    except TypeError:
        pass

    raise TypeError(
        f"Invalid target type or value: {target}"
        "Expected a type or value that is compatible with the 'output_type' parameter"
        "when running a Pydantic AI agent."
        "Please provide a valid type or value."
        "Examples:"
        "- A simple type (str, int, float, bool, Literal, etc.)"
        "- A BaseModel type / subclass"
        "- A dataclass"
        "- A dictionary"
    )


def _process_generic_arg(
    arg: Any,
    make_optional: bool = True,
) -> Any:
    """
    Process a generic type argument, recursively converting nested BaseModels
    to partial versions while handling Union, List, Dict, etc.
    """
    import sys
    import types

    # Support for Union types across Python versions
    if sys.version_info >= (3, 10):
        UNION_ORIGINS = (Union, types.UnionType)
    else:
        UNION_ORIGINS = (Union,)

    arg_origin = get_origin(arg)

    if arg_origin is not None:
        # Handle nested generic types (Union, List, Dict, etc.)
        nested_args = get_args(arg)
        modified_nested_args = tuple(
            _process_generic_arg(t, make_optional=make_optional)
            for t in nested_args
        )

        # Special handling for Union types
        if arg_origin in UNION_ORIGINS:
            return Union[modified_nested_args]

        return arg_origin[modified_nested_args]
    else:
        # Check if it's a BaseModel that needs to be made partial
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            # Prevent infinite recursion for self-referential models
            if arg in _processing_models:
                return arg

            _processing_models.add(arg)
            try:
                return partial_output_model(arg, make_optional=make_optional)
            finally:
                _processing_models.discard(arg)
        else:
            return arg


def _make_field_optional(
    field_info: Any,  # FieldInfo from Pydantic
    make_optional: bool = True,
) -> tuple[Any, Any]:
    """
    Convert a Pydantic field to be optional, handling nested models and generics.

    Args:
        field_info: The Pydantic FieldInfo object
        make_optional: Whether to make the field optional (defaults to True)

    Returns:
        Tuple of (annotation, field_info) for use with create_model
    """
    from copy import deepcopy

    tmp_field = deepcopy(field_info)
    annotation = field_info.annotation

    if not make_optional:
        # If not making optional, just process nested models
        if get_origin(annotation) is not None:
            generic_base = get_origin(annotation)
            generic_args = get_args(annotation)
            modified_args = tuple(
                _process_generic_arg(arg, make_optional=False)
                for arg in generic_args
            )
            tmp_field.annotation = (
                generic_base[modified_args] if generic_base else None
            )
        elif isinstance(annotation, type) and issubclass(
            annotation, BaseModel
        ):
            if annotation not in _processing_models:
                _processing_models.add(annotation)
                try:
                    tmp_field.annotation = partial_output_model(
                        annotation, make_optional=False
                    )
                finally:
                    _processing_models.discard(annotation)

        return tmp_field.annotation, tmp_field

    # Handle generics (List, Dict, Union, etc.)
    if get_origin(annotation) is not None:
        generic_base = get_origin(annotation)
        generic_args = get_args(annotation)

        modified_args = tuple(
            _process_generic_arg(arg, make_optional=True)
            for arg in generic_args
        )

        # Make the entire field Optional
        tmp_field.annotation = (
            Optional[generic_base[modified_args]] if generic_base else None
        )
        tmp_field.default = None
        tmp_field.default_factory = None

    # Handle nested BaseModel fields
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in _processing_models:
            # Already processing this model, avoid infinite recursion
            tmp_field.annotation = Optional[annotation]
        else:
            _processing_models.add(annotation)
            try:
                partial_model = partial_output_model(
                    annotation, make_optional=True
                )
                tmp_field.annotation = Optional[partial_model]
            finally:
                _processing_models.discard(annotation)

        tmp_field.default = None
        tmp_field.default_factory = None

    # Handle simple types
    else:
        tmp_field.annotation = Optional[annotation]
        tmp_field.default = None
        tmp_field.default_factory = None

    return tmp_field.annotation, tmp_field


def partial_output_model(
    model: Type[BaseModel],
    make_optional: bool = True,
) -> Type[BaseModel]:
    """Creates a partial version of a Pydantic BaseModel by optionally making all
    fields optional and recursively processing nested models.

    This is useful for streaming/incremental validation where you want to accept
    incomplete data and validate it progressively.

    Args:
        model: The BaseModel class to make partial
        make_optional: If True, all fields become Optional with None defaults.
                      If False, just wraps nested models without changing optionality.

    Returns:
        A new BaseModel class with the same structure but optional fields

    Example:
        ```python
        class User(BaseModel):
            name: str
            age: int
            email: str

        PartialUser = make_target_partial(User)
        # PartialUser fields: name: Optional[str], age: Optional[int], email: Optional[str]

        # Can create with incomplete data
        user = PartialUser(name="John")  # age and email are None
        ```

    Note:
        - Handles nested BaseModels recursively
        - Handles generic types (List, Dict, Union, etc.)
        - Prevents infinite recursion for self-referential models
        - Original model reference is stored in _original_model attribute
    """
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError(f"Expected a BaseModel type, got {model}")

    # Generate name for partial model
    model_name = (
        model.__name__
        if model.__name__.startswith("Partial")
        else f"Partial{model.__name__}"
    )

    # Process each field
    processed_fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_info in model.model_fields.items():
        annotation, field = _make_field_optional(
            field_info, make_optional=make_optional
        )
        processed_fields[field_name] = (annotation, field)

    # Create the partial model
    partial_model = create_model(
        model_name,
        __base__=model,
        __module__=model.__module__,
        **processed_fields,
    )  # type: ignore[call-overload]

    # Store reference to original model for validation purposes
    partial_model._original_model = model

    return partial_model


def selection_output_model(
    options: list | type,
    name: str | None = None,
    literal: bool = False,
    reason: bool = False,
) -> Type[BaseModel]:
    """
    Helper for generating a model used to select an index from a set of options.
    Used within the `select` semantic operation.

    Accepts:
    - list[...]
    - Literal[...]
    - Enum
    - Union[..., ...]

    Args:
        options: List of options or a type (Literal/Enum/Union).
        name: Optional name for the model. Defaults to "Selection".
        literal: Whether to use a Literal[...] type for the `index` field.
                 (Only if all options are of the same basic type: str/int/float/bool.)
        reason: Include an optional `reason` field.

    Returns:
        A Pydantic BaseModel with an `index` (or choice) and optional `reason`.
    """
    from typing import Literal, get_origin, get_args
    from pydantic import Field, create_model

    # Helper to flatten and extract options just like _collect_select_options
    def _extract_choices(options):
        origin = get_origin(options)
        if isinstance(options, list):
            return options
        if origin is not None:
            if origin is list:
                return list(get_args(options))
        if origin is not None and hasattr(origin, "__members__"):
            return [member for member in options]
        if origin is not None:
            # Union / Literal
            return list(get_args(options))
        if isinstance(options, type) and hasattr(options, "__members__"):
            return list(options)
        raise TypeError("options must be list, Literal, Enum, or Union")

    choices = _extract_choices(options)
    _model_name = name or "Selection"

    doc_choices = "\nChoices:\n" + "\n".join(
        f"- {repr(choice)}" for choice in choices
    )
    doc = (
        f"Structured selection model. Select by index from given options.\n"
        f"{doc_choices}\n"
        "Fields:\n"
        "- index: (int or Literal) the index of the selected option."
        + (
            "\n- reason: (str, optional) explanation for the selection."
            if reason
            else ""
        )
    )

    # Select the field type for the 'index'
    # Literal mode: only if enabled and all choices are the same primitive type
    literal_type = None
    if (
        literal
        and choices
        and all(
            type(c) is type(choices[0]) and type(c) in (int, str, float, bool)
            for c in choices
        )
    ):
        literal_type = Literal[tuple(choices)]  # type: ignore[literal-required]
        index_field = (
            literal_type,
            Field(..., description="Selected option value (literal)"),
        )
    else:
        index_field = (
            int,
            Field(..., description="Selected option index (integer)"),
        )

    # Add optional reason if required
    fields = {"index": index_field}
    if reason:
        fields["reason"] = (
            str | None,
            Field(None, description="Explanation or reason for selection"),
        )

    # Create model, set docstring
    model = create_model(_model_name, **fields, __doc__=doc)  # type: ignore[call-overload]

    # Store the original choices for reference/use
    model._choices = choices

    return model


def sparse_output_model(
    source: Type[BaseModel | Any],
    name: str | None = None,
) -> Type[BaseModel]:
    """Builds a structured output model/schema for *sparse* changes to
    fields.

    Unlike the ``partial_output_model`` function, this function creates
    a ``changes``-list schema, where each entry is a typed, discriminated
    variant::

        class Updates(BaseModel):
            changes: list[Change_name | Change_age | ...]

    Each ``Change_<field>`` variant carries:

    * ``field`` — ``Literal["<name>"]``, used as the discriminator.
    * ``value`` — the *original* field type, fully typed per-field.

    The model only needs to generate entries for the fields it actually
    modifies, saving tokens and making the intent unambiguous.

    Args:
        source : Type[BaseModel | Any]
            The source model or type to build a selective output model for.
        name : str | None = None
            The name of the model. If not provided, the name 'Changes' will be used.

    Returns:
        A new BaseModel class with the same structure but optional fields
    """
    change_variants: list[Type[BaseModel]] = []

    for field, info in source.model_fields.items():
        # NOTE:
        # fields default to str if they dont contain a type annotation (probably impossible in this
        # context), but to standardize llm's response to the field.
        annotation = info.annotation if info.annotation is not None else str

        variant = create_model(
            f"update_{field}",
            field=(Literal[field], field),  # type: ignore[valid-type]
            value=(annotation, ...),
        )
        change_variants.append(variant)

    if not change_variants:
        return source

    union = (
        Union[tuple(change_variants)]
        if len(change_variants) > 1
        else change_variants[0]
    )
    return create_model(
        name or "Updates",
        changes=(
            list[Annotated[union, Field(discriminator="field")]],  # type: ignore[valid-type]
            ...,
        ),
    )


def split_output_model(
    model: Type[BaseModel],
) -> dict[str, Type[BaseModel]]:
    """Splits a BaseModel into separate models for each field, with field content
    normalized to a "content" attribute.

    Args:
        model : Type[BaseModel]
            The BaseModel class to split.

    Returns:
        A dictionary mapping original field names to their corresponding models.
    """
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError(f"Expected a BaseModel type, got {model}")

    result: dict[str, Type[BaseModel]] = {}

    for field_name, field_info in model.model_fields.items():
        # Capitalize field name for model name
        model_name = field_name.capitalize()

        # Get field annotation and metadata
        annotation = field_info.annotation

        # Extract description from field for use as docstring
        description = None
        if hasattr(field_info, "description") and field_info.description:
            description = field_info.description
        elif hasattr(field_info, "title") and field_info.title:
            description = field_info.title

        # Prepare the "content" field with same type and constraints
        # Preserve all field metadata
        from copy import deepcopy

        content_field_info = deepcopy(field_info)

        # Create the new model with "content" field
        split_model = create_model(
            model_name,
            __module__=model.__module__,
            content=(annotation, content_field_info),
        )

        # Add docstring if description exists
        if description:
            split_model.__doc__ = description

        result[field_name] = split_model

    return result
