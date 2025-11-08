"""zyx.core.schemas.schema"""

from __future__ import annotations

import inspect
from copy import deepcopy
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Generic,
    ParamSpec,
    Type,
    TypeVar,
    overload,
)

from pydantic import BaseModel

from ..exceptions import SchemaExecutionError, SchemaValidationError
from . import utils

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

__all__ = ["Schema", "to_schema"]


T = TypeVar("T")
P = ParamSpec("P")
T_co = TypeVar("T_co", covariant=True)


class Schema(Generic[T]):
    """'Universal' & base representation of a schema that can be created
    from any python type or value. This class utilizes various resources
    from Pydantic's JSON schema capabilities, and instructor's OpenAI
    function schema generation capabilities.

    Tools are a subclass of Schema.

    Can be used as:
        - Schema(int) - create schema for a type
        - Schema[int] - create schema for a type (class getitem syntax)
        - Schema(my_function) - create schema for a function
        - schema.validate(data) - validate data
        - schema(data) - validate data (if not a function) or call function (if function)
    """

    @property
    def source(self) -> Type[T] | Callable[..., T]:
        """The source type (or callable) this schema was created from. A schema's source is
        immutable after creation."""
        return self._source

    @property
    def is_function(self) -> bool:
        """Check if this schema wraps a function."""
        return inspect.isfunction(self._source) or inspect.ismethod(self._source)

    def __init__(
        self,
        source: T | Type[T] | Callable[P, T],
        *,
        title: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
        key: str | None = None,
    ) -> None:
        """Create a schema from a source object, type, or function. A schema
        is exactly what it sounds like, a schema for representation of the
        given source. A schema can be accessed as a validator, a pydantic model
        and the OpenAI tool calling format.

        Parameters
        ----------
        source : T | Type[T] | Callable[P, T]
            The source to create a schema from. Can be:
            - A type (e.g., int, str, MyClass)
            - An instance (will use its type)
            - A function or method
        title : str | None
            Optional custom name for the schema
        description : str | None
            Optional custom description for the schema
        exclude : set[str] | None
            Set of parameter names to exclude (for function schemas)
        key: str | None
            For types such as int, Literal, etc. Instructor assigns the key of the field to use
            the name `content`. You can override this value for these simple type cases
            by providing a custom key.

        Returns
        -------
        Schema[T]
            A Schema instance wrapping the source

        Examples:
        --------
            ```python
            >>> # Create schema from a type
            >>> int_schema = Schema(int)
            >>> int_schema.validate(42)
            42

            >>> # Create schema from a function
            >>> @Schema
            ... def search(query: str, limit: int = 10) -> list[str]:
            ...     return [f"result for {query}"]
            >>> search(query="python")
            ['result for python']

            >>> # Create schema with exclusions
            >>> def api_call(query: str, api_key: str) -> dict:
            ...     pass
            >>> api_schema = Schema(api_call, exclude={"api_key"})
            ```
        """
        if inspect.isfunction(source) or inspect.ismethod(source):
            self._source = source
        else:
            # Check if it's a typing construct (like Literal, Union, etc.)
            # These have __origin__ attribute
            if hasattr(source, "__origin__") or hasattr(source, "__args__"):
                # It's a typing construct, use as-is
                self._source = source
            elif not isinstance(source, type):
                # It's an instance, get its type
                self._source = type(source)
            else:
                # It's a regular type
                self._source = source

        # mutable
        self.title = title
        self.description = description
        self.exclude = exclude or set()
        self.key = key

    # this is cached in utils instead of here because
    # i wanted the ide hinting to show up like a type, not a value
    # when using @cached_property the hinting shows as a value, but
    # @property is able to retain this type hinting. (basedpyright)
    @property
    def pydantic_model(self) -> Type[BaseModel]:
        """A Pydantic model representation of this object, function, or type.

        For functions, this returns a model of the function's parameters (excluding any excluded fields).
        For types, returns a model that can validate that type (excluding any excluded fields).
        For BaseModel, returns it with excluded fields filtered out.

        Note: The exclude set is applied at model creation time.
        """
        model = utils.create_pydantic_model(self.source, name=self.title, key=self.key)

        # If there are exclusions, filter the model
        if self.exclude:
            from pydantic import create_model

            # Get fields that aren't excluded
            filtered_fields = {
                k: (v.annotation, v.default if v.default is not ... else ...)
                for k, v in model.model_fields.items()
                if k not in self.exclude
            }

            return create_model(
                model.__name__,
                __doc__=model.__doc__,
                **filtered_fields,  # type: ignore
            )

        return model

    @cached_property
    def json_schema(self) -> utils.ObjectJsonSchema:
        """A JSON schema representation of this object, function, or type.

        For functions, this returns the parameter schema.
        For types, returns the JSON schema.

        This is derived from the Pydantic model - EVERYTHING goes through that pipeline.
        """
        # Get the JSON schema from the Pydantic model
        return self.pydantic_model.model_json_schema()

    @cached_property
    def openai_schema(self) -> "ChatCompletionToolParam":
        """An OpenAI tool representation - only available for function schemas.

        This is the base schema without any exclusions.
        Use render_openai_schema() to customize with exclusions.
        """
        if not self.is_function:
            raise AttributeError(
                "openai_schema is only available for function schemas. "
                "Use json_schema or pydantic_model for types."
            )

        return utils.openai_function_schema(
            self.source,  # type: ignore
            name=self.title,
            description=self.description,
        )

    def render_openai_schema(
        self,
        exclude: set[str] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> "ChatCompletionToolParam":
        """Render an OpenAI tool schema with optional customizations.

        Parameters
        ----------
        exclude : set[str] | None
            Additional parameter names to exclude (merged with init-level excludes).
        name : str | None
            Optional override for the function name.
        description : str | None
            Optional override for the function description.

        Returns
        -------
        ChatCompletionToolParam
            The OpenAI tool schema with the optional customizations.

        Raises
        ------
        AttributeError: If called on a non-function schema.

        Examples
        --------
            ```python
            >>> def search(query: str, api_key: str, debug: bool) -> list[str]: ...
            >>> # Exclude at init
            >>> schema = Schema(search, exclude={"api_key"})
            >>> # Exclude additional params at render time
            >>> schema.render_openai_schema(exclude={"debug"})
            >>> # Result: both api_key and debug are excluded
            ```
        """
        if not self.is_function:
            raise AttributeError(
                "render_openai_schema is only available for function schemas. "
                "Use json_schema or pydantic_model for types."
            )

        # Start with the base schema (make a copy to avoid mutating cached version)
        schema = deepcopy(self.openai_schema)
        func_schema = schema["function"]  # type: ignore

        # Override name if provided
        if name:
            func_schema["name"] = name

        # Override description if provided
        if description:
            func_schema["description"] = description

        # Merge init-level and runtime exclusions
        all_exclusions = self.exclude | (exclude or set())

        # Apply exclusions if any
        if all_exclusions:
            params = func_schema["parameters"]
            properties = params.get("properties", {})
            required = params.get("required", [])

            # Filter out excluded parameters
            func_schema["parameters"] = {
                "type": "object",
                "properties": {
                    k: v for k, v in properties.items() if k not in all_exclusions
                },
                "required": [r for r in required if r not in all_exclusions],
            }

        return schema

    def validate(
        self,
        data: Any,
        *,
        raise_on_error: bool = True,
    ) -> T | dict[str, Any]:
        """Validate some given data against this schema.

        Parameters
        ----------
        data : Any
            The data to validate (dict, primitive, or any JSON-serializable value)
        raise_on_error : bool
            If False, returns error dict instead of raising exceptions

        Returns
        -------
        T | dict[str, Any]
            The validated and typed result, or error dict if raise_on_error=False

        Raises
        ------
        TypeError: If called on a function schema.
        SchemaValidationError: If validation fails and raise_on_error=True

        Examples
        --------
        ```python
            >>> schema = Schema(int)
            >>> schema.validate(42)
            42

            >>> schema = Schema(User)
            >>> schema.validate({"name": "John", "age": 30})
            User(name='John', age=30)

            >>> schema.validate("invalid", raise_on_error=False)
            {'error': '...', 'type': 'ValidationError'}
        """
        if self.is_function:
            raise TypeError(
                "Cannot validate data for a function schema. "
                "Use __call__ to execute the function with validated parameters."
            )

        if raise_on_error:
            try:
                return utils.validate_with_schema(data, self.source)  # type: ignore
            except Exception as e:
                raise SchemaValidationError(
                    str(e), [{"field": "root", "msg": str(e)}]
                ) from e
        else:
            try:
                return utils.validate_with_schema(data, self.source)  # type: ignore
            except Exception as e:
                return {
                    "error": str(e),
                    "type": type(e).__name__,
                }

    def execute(
        self,
        params: dict[str, Any],
        *,
        use_defaults: bool = True,
        raise_on_error: bool = True,
    ) -> Any:
        """Execute a function schema with the provided parameters.

        This method is only available for function schemas and provides
        fine-grained control over execution behavior.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of parameter names to values.
        use_defaults : bool
            If True, uses default values for missing parameters.
        raise_on_error : bool
            If False, returns error dict instead of raising exceptions.

        Returns
        -------
        Any
            The result of the function execution, or error dict if raise_on_error=False.

        Raises
        ------
        TypeError: If called on a non-function schema.
        SchemaExecutionError: If validation fails and raise_on_error=True.

        Examples
        --------
            >>> def search(query: str, limit: int = 10) -> list[str]:
            ...     return [f"result for {query}"]
            >>> schema = Schema(search)
            >>> schema.execute({"query": "python"})
            ['result for python']
            >>> schema.execute({"query": "python", "limit": 5})
            ['result for python']
            >>> # Error handling
            >>> schema.execute({"invalid": "param"}, raise_on_error=False)
            {'error': '...', 'type': 'TypeError'}
        """
        if not self.is_function:
            raise TypeError(
                "execute() is only available for function schemas. "
                "Use validate() for type schemas."
            )

        try:
            sig = inspect.signature(self._source)  # type: ignore

            # Build arguments with defaults if needed
            bound_args = {}
            for param_name, param in sig.parameters.items():
                if param_name in params:
                    # Parameter provided, validate it
                    if param.annotation is not inspect.Parameter.empty:
                        try:
                            bound_args[param_name] = utils.validate_with_schema(
                                params[param_name], param.annotation
                            )
                        except Exception as e:
                            if raise_on_error:
                                raise TypeError(
                                    f"Parameter '{param_name}' validation failed: {e}"
                                ) from e
                            return {
                                "error": f"Parameter '{param_name}' validation failed: {e}",
                                "type": "ValidationError",
                            }
                    else:
                        bound_args[param_name] = params[param_name]

                elif use_defaults and param.default is not inspect.Parameter.empty:
                    # Use default value
                    bound_args[param_name] = param.default

                elif param.default is inspect.Parameter.empty:
                    # Required parameter missing
                    if raise_on_error:
                        raise SchemaExecutionError(
                            f"Missing required parameter: {param_name}",
                            [
                                {
                                    "field": param_name,
                                    "msg": "Missing required parameter",
                                }
                            ],
                        )
                    return {
                        "error": f"Missing required parameter: {param_name}",
                        "type": "TypeError",
                    }

            # Execute the function
            return self._source(**bound_args)  # type: ignore

        except Exception as e:
            if raise_on_error:
                raise
            return {
                "error": str(e),
                "type": type(e).__name__,
            }

    def copy(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
    ) -> Schema[T]:
        """Copy the schema with optional customizations.

        Parameters
        ----------
        name : str | None
            Optional override for the function name.
        description : str | None
            Optional override for the function description.
        exclude : set[str] | None
            Optional override for the exclude set.

        Returns
        -------
        Schema[T]
            A copy of the schema with the optional customizations.
        """
        return Schema(
            self.source,
            name=name or self.title,
            description=description or self.description,
            exclude=exclude if exclude is not None else self.exclude,
        )

    @overload
    def __call__(
        self: Schema[Callable[P, T_co]], *args: P.args, **kwargs: P.kwargs
    ) -> T_co:
        """For sync functions: validate params and execute."""
        ...

    @overload
    def __call__(
        self: Schema[Callable[P, Coroutine[Any, Any, T_co]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Coroutine[Any, Any, T_co]:
        """For async functions: validate params and execute."""
        ...

    @overload
    def __call__(self, data: Any | T) -> T:
        """For types: validate the data."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the schema - behavior depends on whether it's a function or type.

        For functions:
            - Validates the parameters and executes the function
            - Returns the function's result

        For types:
            - Alias for validate() - validates the input data
            - Returns the validated data

        Examples:
        --------
            ```python
            >>> # Type schema - validates data
            >>> int_schema = Schema[int]
            >>> int_schema(42)
            42
            >>> int_schema("invalid")  # raises ValidationError

            >>> # Function schema - validates params and executes
            >>> def search(query: str, limit: int = 10) -> list[str]:
            ...     return [f"result for {query}"]
            >>> search_schema = Schema[search]
            >>> search_schema(query="python", limit=5)
            ['result for python']
            >>> search_schema(query=123)  # raises ValidationError (wrong type)
            ```
        """
        if self.is_function:
            # For functions: validate parameters and execute
            sig = inspect.signature(self._source)

            # Bind the arguments to the function signature
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError as e:
                raise TypeError(
                    f"Invalid arguments for {self._source.__name__}: {e}"
                ) from e

            # Validate each parameter against its type annotation
            validated_params = {}
            for param_name, param_value in bound.arguments.items():
                param = sig.parameters[param_name]

                # If the parameter has a type annotation, validate against it
                if param.annotation is not inspect.Parameter.empty:
                    try:
                        validated_params[param_name] = utils.validate_with_schema(
                            param_value, param.annotation
                        )
                    except Exception as e:
                        raise TypeError(
                            f"Parameter '{param_name}' validation failed: {e}"
                        ) from e
                else:
                    # No type annotation, pass through as-is
                    validated_params[param_name] = param_value

            # Execute the function with validated parameters
            return self._source(**validated_params)  # type: ignore
        else:
            # For types: validate the input (use first arg or kwargs)
            if args and kwargs:
                raise TypeError(
                    "Provide either positional or keyword arguments, not both"
                )

            if args:
                if len(args) != 1:
                    raise TypeError(f"Expected 1 argument, got {len(args)}")
                return self.validate(args[0])
            elif kwargs:
                return self.validate(kwargs)
            else:
                raise TypeError("No data provided to validate")

    def __repr__(self) -> str:
        source_name = (
            self.source.__name__
            if hasattr(self.source, "__name__")
            else str(self.source)
        )
        schema_type = "function" if self.is_function else "type"
        return f"Schema({schema_type}={source_name}, name={self.title!r})"


def to_schema(
    source: T | Type[T] | Callable[P, T],
    *,
    title: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
    key: str | None = None,
) -> Schema[T]:
    """Create a schema from a source object, type, or function. A schema
    is exactly what it sounds like, a schema for representation of the
    given source. A schema can be accessed as a validator, a pydantic model
    and the OpenAI tool calling format.

    Parameters
    ----------
    source : T | Type[T] | Callable[P, T]
        The source to create a schema from. Can be:
        - A type (e.g., int, str, MyClass)
        - An instance (will use its type)
        - A function or method
    title : str | None
        Optional custom title for the schema
    description : str | None
        Optional custom description for the schema
    exclude : set[str] | None
        Set of parameter names to exclude (for function schemas)
    key: str | None
        For types such as int, Literal, etc. Instructor assigns the key of the field to use
        the name `content`. You can override this value for these simple type cases
        by providing a custom key.

    Returns
    -------
    Schema[T]
        A Schema instance wrapping the source

    Examples:
    --------
        ```python
        >>> # Create schema from a type
        >>> int_schema = to_schema(int)
        >>> int_schema.validate(42)
        42

        >>> # Create schema from a function
        >>> @to_schema
        ... def search(query: str, limit: int = 10) -> list[str]:
        ...     return [f"result for {query}"]
        >>> search(query="python")
        ['result for python']

        >>> # Create schema with exclusions
        >>> def api_call(query: str, api_key: str) -> dict:
        ...     pass
        >>> api_schema = to_schema(api_call, exclude={"api_key"})
        ```
    """
    return Schema(source, title=title, description=description, exclude=exclude)
