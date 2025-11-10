"""zyx.core.processing.schemas.semantics"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Type, TypeVar, Union, get_args, get_origin

from typing_inspect import is_literal_type, is_optional_type, is_union_type

from ...._internal._exceptions import ProcessingError

__all__ = (
    "to_semantic_title",
    "to_semantic_key",
    "to_semantic_description",
)


T = TypeVar("T")


_SINGULAR_TITLES: dict[type, str] = {
    int: "Integer",
    float: "Number",
    bool: "Boolean",
    str: "String",
    bytes: "Bytes",
    list: "List",
    tuple: "Tuple",
    dict: "Dictionary",
    set: "Set",
    frozenset: "FrozenSet",
}
"""Singular type names for model titles."""


_PLURAL_TITLES: dict[type, str] = {
    int: "Integers",
    float: "Numbers",
    bool: "Booleans",
    str: "Strings",
    bytes: "ByteArrays",
    list: "Lists",
    tuple: "Tuples",
    dict: "Dictionaries",
    set: "Sets",
    frozenset: "FrozenSets",
}
"""Plural type names for collections."""


_SEMANTIC_KEYS: dict[type, str] = {
    int: "value",
    float: "value",
    bool: "flag",
    str: "text",
    bytes: "data",
    list: "items",
    tuple: "values",
    dict: "mapping",
    set: "elements",
    frozenset: "elements",
}
"""Semantic field names based on type."""


def _is_numeric_type(t: type) -> bool:
    """Check if a type is numeric (int or float)."""
    return t in (int, float)


def _is_collection_type(t: type) -> bool:
    """Check if a type is a collection (list, set, tuple, frozenset)."""
    return t in (list, tuple, set, frozenset)


def _get_inner_type(t: Any) -> Any:
    """Extract the inner type from Optional or generic types."""
    origin = get_origin(t)
    args = get_args(t)

    if origin is Union:
        # For Optional[T] (Union[T, None]), return T
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]

    # For generic types like List[T], return T
    if args:
        return args[0]

    return None


def to_semantic_title(t: Type[T] | T) -> str:
    """
    Generate a semantic, human-readable title for a type.

    This function produces natural model names that reflect the semantic
    meaning of the type rather than mechanical concatenation. It handles:

    - Simple types: int → "Integer", str → "String"
    - Collections: list[int] → "Integers", list[str] → "Strings"
    - Optionals: Optional[int] → "OptionalInteger"
    - Unions: Union[str, int] → "Choice" (when distinct types)
    - Literals: Literal["a", "b"] → "Selection"
    - Complex generics: dict[str, int] → "StringToInteger"
    - Dict objects: {"name": str, "age": int} → "Data"

    Parameters
    ----------
    t: Type[T] | dict
        The type hint to generate a semantic title for.
        Can also be a dict object for field definitions.

    Returns
    -------
    str: A semantic title suitable for a model name

    Raises
    ------
    ProcessingError
        If the type cannot be processed or is invalid.

    Examples:
    --------
        ```python
        >>> to_semantic_title(int)
        'Integer'

        >>> to_semantic_title(list[str])
        'Strings'

        >>> to_semantic_title(dict[str, int])
        'StringToInteger'

        >>> to_semantic_title({"name": str, "age": int})
        'Data'

        >>> to_semantic_title(Literal["red", "green", "blue"])
        'Selection'

        >>> to_semantic_title(Union[int, str])
        'Choice'

        >>> to_semantic_title(Optional[str])
        'OptionalString'
        ```
    """
    # Handle dict objects (field definitions) - not hashable
    if isinstance(t, dict):
        return "Data"

    # Use cached internal implementation for hashable types
    @lru_cache(maxsize=512)
    def _cached(type_obj: Any) -> str:
        if type_obj is None:
            raise ProcessingError(
                "Cannot generate semantic title for None. "
                "Use type(None) or NoneType instead."
            )

        # Handle NoneType
        if type_obj is type(None):
            return "Null"

        # Handle basic types (singular)
        if type_obj in _SINGULAR_TITLES:
            return _SINGULAR_TITLES[type_obj]

        origin = get_origin(type_obj)
        args = get_args(type_obj)

        # Handle Literal types - these represent choices/selections
        if is_literal_type(type_obj):
            return "Selection"

        # Handle Optional types
        if is_optional_type(type_obj):
            inner = _get_inner_type(type_obj)
            if inner:
                inner_title = _cached(inner)
                return f"Optional{inner_title}"
            return "OptionalValue"

        if is_union_type(type_obj):
            non_none_args = [arg for arg in args if arg is not type(None)]

            # If all types are the same category (e.g., all numeric), unify
            if len(non_none_args) == 2:
                if all(
                    _is_numeric_type(arg)
                    for arg in non_none_args
                    if isinstance(arg, type)
                ):
                    return "NumericChoice"

            # For distinct types, call it a Choice
            if len(non_none_args) > 1:
                return "Choice"

            # Shouldn't reach here, but handle gracefully
            return "Value"

        # Handle generic types with arguments
        if origin is not None:
            # List, Set, FrozenSet - use plural form of element type
            if origin in (list, set, frozenset):
                if args:
                    element_type = args[0]

                    # For simple types, use plural
                    if element_type in _PLURAL_TITLES:
                        return _PLURAL_TITLES[element_type]

                    # For complex types, append 'List', 'Set', etc.
                    element_title = _cached(element_type)
                    origin_name = _SINGULAR_TITLES.get(
                        origin, origin.__name__.capitalize()
                    )
                    return f"{element_title}{origin_name}"

                # No args, use base name
                return _SINGULAR_TITLES.get(
                    origin, origin.__name__.capitalize()
                )

            # Dict - use 'KeyToValue' format for semantic clarity
            elif origin is dict:
                if args and len(args) >= 2:
                    key_type = args[0]
                    value_type = args[1]

                    key_title = _cached(key_type)
                    value_title = _cached(value_type)

                    # Use natural 'To' connector
                    return f"{key_title}To{value_title}"

                return "Dictionary"

            # Tuple - handle as a group of values
            elif origin is tuple:
                if args:
                    # For homogeneous tuples, use plural + 'Pair', 'Triple', etc.
                    if len(set(args)) == 1 and args[0] != Ellipsis:
                        element_title = _cached(args[0])
                        count_name = {
                            2: "Pair",
                            3: "Triple",
                            4: "Quadruple",
                        }.get(len(args), "Tuple")
                        return f"{element_title}{count_name}"

                    # For heterogeneous tuples, just call it a Tuple
                    return "Tuple"

                return "Tuple"

            # Other generic types - use the origin name
            origin_name = _SINGULAR_TITLES.get(
                origin, getattr(origin, "__name__", "Generic")
            )
            if origin_name:
                if args:
                    # Prepend the first arg type for context
                    first_arg = _cached(args[0])
                    return f"{first_arg}{origin_name}"
                return origin_name

        # Handle custom classes and unknown types
        if hasattr(type_obj, "__name__"):
            name = type_obj.__name__
            if not name:
                raise ProcessingError(
                    f"Type {type_obj!r} has an empty __name__ attribute. "
                    "Cannot generate semantic title."
                )
            # Capitalize and return
            return name if name[0].isupper() else name.capitalize()

        # Absolute fallback - this should rarely be reached
        raise ProcessingError(
            f"Cannot generate semantic title for type {type_obj!r}. "
            "Type must have a __name__ attribute or be a recognized type hint."
        )

    return _cached(t)


def to_semantic_key(t: Type[T] | T) -> str:
    """
    Generate a semantic, context-appropriate field name for a type.

    This function produces natural field/variable names that reflect
    how the data would be used in practice. It handles:

    - Simple types: int → "value", str → "text"
    - Collections: list[Any] → "items", dict[Any, Any] → "mapping"
    - Optionals: Optional[str] → "text" (uses inner type)
    - Unions: Union[str, int] → "choice"
    - Literals: Literal["a", "b"] → "selection"
    - Dict objects: {"name": str, "age": int} → "content"

    Parameters
    ----------
    t: Type[T] | dict
        The type hint to generate a semantic key for.
        Can also be a dict object for field definitions.

    Returns
    -------
    str: A semantic field name suitable for a variable or parameter

    Raises
    ------
    ProcessingError
        If the type cannot be processed or is invalid.

    Examples:
    --------
        ```python
        >>> to_semantic_key(int)
        'value'

        >>> to_semantic_key(str)
        'text'

        >>> to_semantic_key(list[str])
        'items'

        >>> to_semantic_key(dict[str, int])
        'mapping'

        >>> to_semantic_key({"name": str, "age": int})
        'content'

        >>> to_semantic_key(Literal["option1", "option2"])
        'selection'

        >>> to_semantic_key(Union[int, str])
        'choice'

        >>> to_semantic_key(Optional[str])
        'text'
        ```
    """
    # Handle dict objects (field definitions) - not hashable
    if isinstance(t, dict):
        return "content"

    # Use cached internal implementation for hashable types
    @lru_cache(maxsize=512)
    def _cached(type_obj: Any) -> str:
        if type_obj is None:
            raise ProcessingError(
                "Cannot generate semantic key for None. "
                "Use type(None) or NoneType instead."
            )

        # Handle NoneType
        if type_obj is type(None):
            return "value"

        # Handle basic types
        if type_obj in _SEMANTIC_KEYS:
            return _SEMANTIC_KEYS[type_obj]

        origin = get_origin(type_obj)
        args = get_args(type_obj)

        if is_literal_type(type_obj):
            return "selection"

        # Handle Optional types - use the inner type's key
        if is_optional_type(type_obj):
            inner = _get_inner_type(type_obj)
            if inner:
                return _cached(inner)
            return "value"

        # Handle Union types - these represent a choice
        if origin is Union:
            non_none_args = [arg for arg in args if arg is not type(None)]

            # If multiple distinct types, it's a choice
            if len(non_none_args) > 1:
                return "choice"

            # Single type (shouldn't happen, but handle it)
            if len(non_none_args) == 1:
                return _cached(non_none_args[0])

            return "value"

        # Handle generic types with arguments
        if origin is not None:
            # Use the origin's semantic key if available
            if origin in _SEMANTIC_KEYS:
                return _SEMANTIC_KEYS[origin]

            # Special handling for specific origins
            if origin is list:
                return "items"
            elif origin is tuple:
                return "values"
            elif origin is dict:
                return "mapping"
            elif origin in (set, frozenset):
                return "elements"

        # Handle custom classes - use lowercase class name
        if hasattr(type_obj, "__name__"):
            name = type_obj.__name__
            if not name:
                raise ProcessingError(
                    f"Type {type_obj!r} has an empty __name__ attribute. "
                    "Cannot generate semantic key."
                )
            # Convert to snake_case-ish and lowercase
            return name.lower()

        # Absolute fallback - this should rarely be reached
        raise ProcessingError(
            f"Cannot generate semantic key for type {type_obj!r}. "
            "Type must have a __name__ attribute or be a recognized type hint."
        )

    return _cached(t)


def to_semantic_description(t: Type[T] | T) -> str:
    """
    Generate a semantic, human-readable description for a type or function.

    This function produces natural descriptions that explain what the type
    or function represents, suitable for documentation and schema descriptions.

    - Simple types: int → "An integer value"
    - Collections: list[str] → "A list of strings"
    - Optionals: Optional[int] → "An optional integer value"
    - Unions: Union[str, int] → "A choice between multiple types"
    - Literals: Literal["a", "b"] → "A selection from predefined options"
    - Functions: func → "Function: <name>" or uses docstring
    - Dict objects: {"name": str} → "A data object"

    Parameters
    ----------
    t: Type[T] | T
        The type hint, function, or object to generate a description for.

    Returns
    -------
    str: A semantic description suitable for documentation

    Raises
    ------
    ProcessingError
        If the type cannot be processed or is invalid.

    Examples:
    --------
        ```python
        >>> to_semantic_description(int)
        'An integer value'

        >>> to_semantic_description(list[str])
        'A list of strings'

        >>> to_semantic_description(Optional[str])
        'An optional string value'

        >>> to_semantic_description({"name": str, "age": int})
        'A data object'

        >>> def search(query: str):
        ...     '''Search for items matching the query.'''
        ...     pass
        >>> to_semantic_description(search)
        'Search for items matching the query.'

        >>> # Examples with mock client methods
        >>> from zyx.core.models.clients.mock import MockModelClient
        >>> client = MockModelClient()
        >>> to_semantic_description(client.chat_completion)
        'Generate a mock chat completion.'

        >>> to_semantic_description(client.structured_output)
        'Generate a mock structured output.'

        >>> to_semantic_description(client.embedding)
        'Generate mock embeddings.'
        ```
    """
    import inspect

    # Handle functions and methods
    if inspect.isfunction(t) or inspect.ismethod(t):
        func_name = getattr(t, "__name__", "function")
        # Try to get docstring first line
        doc = inspect.getdoc(t)
        if doc:
            # Get first non-empty line
            first_line = doc.split("\n")[0].strip()
            if first_line:
                return first_line
        return f"Function: {func_name}"

    # Handle dict objects (field definitions)
    if isinstance(t, dict):
        return "A data object"

    # Handle BaseModel subclasses
    try:
        from pydantic import BaseModel

        if isinstance(t, type) and issubclass(t, BaseModel):
            # Get docstring or class name
            doc = inspect.getdoc(t)
            if doc:
                first_line = doc.split("\n")[0].strip()
                if first_line:
                    return first_line
            return f"Schema for {t.__name__}"
    except (ImportError, TypeError):
        pass

    # Use cached internal implementation for hashable types
    @lru_cache(maxsize=512)
    def _cached(type_obj: Any) -> str:
        if type_obj is None:
            raise ProcessingError(
                "Cannot generate semantic description for None. "
                "Use type(None) or NoneType instead."
            )

        # Handle NoneType
        if type_obj is type(None):
            return "A null value"

        # Handle basic types with articles
        type_descriptions = {
            int: "An integer value",
            float: "A floating-point number",
            bool: "A boolean flag",
            str: "A text string",
            bytes: "Binary data",
            list: "A list of items",
            tuple: "A tuple of values",
            dict: "A dictionary mapping",
            set: "A set of unique elements",
            frozenset: "An immutable set of elements",
        }

        if type_obj in type_descriptions:
            return type_descriptions[type_obj]

        origin = get_origin(type_obj)
        args = get_args(type_obj)

        # Handle Literal types
        if is_literal_type(type_obj):
            values = get_args(type_obj)
            if len(values) <= 3:
                value_str = ", ".join(repr(v) for v in values)
                return f"A selection from: {value_str}"
            return f"A selection from {len(values)} predefined options"

        # Handle Optional types
        if is_optional_type(type_obj):
            inner = _get_inner_type(type_obj)
            if inner:
                inner_desc = _cached(inner)
                # Remove "A" or "An" prefix if present for better grammar
                if inner_desc.startswith("An "):
                    inner_desc = inner_desc[3:]
                elif inner_desc.startswith("A "):
                    inner_desc = inner_desc[2:]
                return f"An optional {inner_desc.lower()}"
            return "An optional value"

        # Handle Union types
        if is_union_type(type_obj):
            non_none_args = [arg for arg in args if arg is not type(None)]

            if len(non_none_args) == 2:
                # Special case for two types
                if all(
                    _is_numeric_type(arg)
                    for arg in non_none_args
                    if isinstance(arg, type)
                ):
                    return "A numeric value (integer or float)"

            if len(non_none_args) > 1:
                return "A choice between multiple types"

            return "A value"

        # Handle generic types with arguments
        if origin is not None:
            # List, Set, FrozenSet
            if origin in (list, set, frozenset):
                container_name = {
                    list: "list",
                    set: "set",
                    frozenset: "immutable set",
                }.get(origin, "collection")

                if args:
                    element_type = args[0]

                    # Get element type description
                    if element_type in type_descriptions:
                        elem_desc = type_descriptions[element_type].lower()
                        # Extract the noun (e.g., "integer value" from "An integer value")
                        if elem_desc.startswith("an "):
                            elem_desc = elem_desc[3:]
                        elif elem_desc.startswith("a "):
                            elem_desc = elem_desc[2:]

                        # Pluralize common types
                        plurals = {
                            "integer value": "integers",
                            "floating-point number": "numbers",
                            "boolean flag": "booleans",
                            "text string": "strings",
                        }
                        elem_desc = plurals.get(elem_desc, elem_desc + "s")

                        return f"A {container_name} of {elem_desc}"

                    # For complex types, use the title
                    try:
                        elem_title = to_semantic_title(
                            element_type
                        ).lower()
                        return f"A {container_name} of {elem_title} items"
                    except:
                        return f"A {container_name} of items"

                return f"A {container_name} of items"

            # Dict
            elif origin is dict:
                if args and len(args) >= 2:
                    key_type = args[0]
                    value_type = args[1]

                    key_name = type_descriptions.get(
                        key_type, str(key_type)
                    ).lower()
                    value_name = type_descriptions.get(
                        value_type, str(value_type)
                    ).lower()

                    # Clean up articles
                    for prefix in ["an ", "a "]:
                        if key_name.startswith(prefix):
                            key_name = key_name[len(prefix) :]
                        if value_name.startswith(prefix):
                            value_name = value_name[len(prefix) :]

                    return f"A mapping from {key_name} to {value_name}"

                return "A dictionary mapping"

            # Tuple
            elif origin is tuple:
                if args:
                    if len(set(args)) == 1 and args[0] != Ellipsis:
                        count_names = {
                            2: "pair",
                            3: "triple",
                            4: "quadruple",
                        }
                        count_name = count_names.get(
                            len(args), f"{len(args)}-tuple"
                        )
                        return f"A {count_name} of values"
                    return f"A tuple of {len(args)} values"
                return "A tuple of values"

        # Handle custom classes
        if hasattr(type_obj, "__name__"):
            name = type_obj.__name__
            if not name:
                raise ProcessingError(
                    f"Type {type_obj!r} has an empty __name__ attribute. "
                    "Cannot generate semantic description."
                )

            # Check if it's a class (type)
            if isinstance(type_obj, type):
                return f"A {name} instance"

            return f"A {name} value"

        # Absolute fallback
        raise ProcessingError(
            f"Cannot generate semantic description for type {type_obj!r}. "
            "Type must have a __name__ attribute or be a recognized type hint."
        )

    return _cached(t)
