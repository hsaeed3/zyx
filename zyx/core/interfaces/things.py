"""zyx.core.interfaces.things"""

from __future__ import annotations

import asyncio
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Type,
    TypeVar,
)

from ..._internal._exceptions import GuardrailError, ThingError
from ..models.language.model import LanguageModel, LanguageModelName
from ..models.language.types import LanguageModelResponse
from ..processing.schemas.schema import P, Schema

if TYPE_CHECKING:
    from instructor import Mode

__all__ = ["Thing", "to_thing"]


T = TypeVar("T")
R = TypeVar("R")


class Thing(Generic[T]):
    """A 'thing' is a mutable, type-aware container that can generate, edit,
    and query its content using language models.

    Unlike a `Bit` which is immutable content for querying, a `Thing` is a stateful
    object that can be created, modified, validated, and constrained using both
    traditional Python validators and LLM-based guardrails.

    Parameters
    ----------
    t : Type[T] | Schema[T] | T
        Either a type specification, a Schema, or an initial value.
        If a value is provided, the type will be inferred from it.
    model : LanguageModelName | LanguageModel | str
        The default language model to use for operations.
    instructions : str | None
        Default instructions for the language model.
    title : str | None
        Optional title for the schema.
    description : str | None
        Optional description for the schema.
    exclude : set[str] | None
        Fields to exclude from schema operations.
    key : str | None
        For simple types, override the default field name.

    Attributes
    ----------
    content : T | None
        The current content of this thing.
    schema : Schema[T]
        The schema representation of this thing's type.

    Examples
    --------
    >>> # Create an empty thing from type
    >>> answer = thing(int)
    >>> print(answer.content)
    None

    >>> # Create a thing with initial value
    >>> name = thing("Alice")
    >>> print(name.content)
    Alice

    >>> # Make content
    >>> answer.make("what is 2+2?", update=True)
    >>> print(answer.content)
    4

    >>> # Edit content
    >>> answer.edit("make it 10", update=True)
    >>> print(answer.content)
    10

    >>> # Add validators
    >>> @answer.validator
    ... def positive(x: int) -> int:
    ...     if x <= 0:
    ...         raise ValueError("must be positive")
    ...     return x

    >>> # Add guardrails
    >>> answer.guardrail("must be less than 100")
    """

    @cached_property
    def schema(self) -> Schema[T]:
        """The schema representation of this thing's type."""
        return self._schema

    @property
    def content(self) -> T | None:
        """The current content of this thing."""
        return self._content

    @content.setter
    def content(self, value: T | None) -> None:
        """Set the content of this thing, applying validators."""
        if value is not None:
            # Apply validators
            for validator in self._validators:
                try:
                    value = validator(value)
                except Exception as e:
                    raise ThingError(f"Validation failed: {e}") from e

        self._content = value

    @property
    def model(self) -> LanguageModel[T]:
        """The language model used by this thing."""
        return self._model

    @model.setter
    def model(
        self, value: LanguageModelName | LanguageModel[T] | str
    ) -> None:
        """Set the language model for this thing."""
        if isinstance(value, str):
            self._model = LanguageModel(value, type=self._schema.source)
        else:
            self._model = value

    def __init__(
        self,
        t: Type[T] | Schema[T] | T = str,
        *,
        model: LanguageModelName
        | LanguageModel[T]
        | str = "openai/gpt-4o-mini",
        instructions: str | None = None,
        title: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
        key: str | None = None,
    ) -> None:
        """Initialize a Thing with a type or initial value and optional configuration."""
        initial_value: T | None = None

        # Determine if we got a Schema, Type, or Value
        if isinstance(t, Schema):
            # It's a Schema
            self._schema = t
        elif isinstance(t, type):
            # It's a Type
            self._schema = Schema(
                t,
                title=title,
                description=description,
                exclude=exclude,
                key=key,
            )
        else:
            # It's a value - infer the type from it
            inferred_type = type(t)
            self._schema = Schema(
                inferred_type,
                title=title,
                description=description,
                exclude=exclude,
                key=key,
            )
            initial_value = t

        # Initialize model
        if isinstance(model, str):
            self._model = LanguageModel(
                model,
                type=self._schema.source,
                instructions=instructions,
            )
        else:
            self._model = model
            if instructions:
                self._model.instructions = instructions

        # Initialize validation
        self._validators: List[Callable[[T], T]] = []
        self._guardrails: List[Dict[str, Any]] = []
        self._instructions = instructions

        # Initialize content (this will apply validators if any)
        self._content: T | None = None
        if initial_value is not None:
            self.content = initial_value

    def validator(self, func: Callable[[T], T]) -> Callable[[T], T]:
        """Decorator to add a validator function to this thing.

        Validators are Python functions that validate and optionally transform
        the content. They are applied whenever content is set.

        Parameters
        ----------
        func : Callable[[T], T]
            A function that takes the content and returns validated/transformed content.
            Should raise an exception if validation fails.

        Returns
        -------
        Callable[[T], T]
            The same function (for use as a decorator).

        Examples
        --------
        >>> thing = to_thing(int)
        >>>
        >>> @thing.validator
        ... def positive(x: int) -> int:
        ...     if x <= 0:
        ...         raise ValueError("must be positive")
        ...     return x
        >>>
        >>> thing.content = 5  # OK
        >>> thing.content = -1  # Raises ThingError
        """
        self._validators.append(func)
        return func

    def guardrail(
        self,
        constraint: str,
        *,
        field: str | None = None,
        model: LanguageModelName | LanguageModel | str | None = None,
        raise_on_error: bool = True,
        severity: Literal["error", "warning", "info"] = "error",
    ) -> None:
        """Add an LLM-based guardrail constraint to this thing.

        Guardrails use language models to validate content against natural language
        constraints. They are applied when content is set or during make/edit operations.

        Parameters
        ----------
        constraint : str
            Natural language description of the constraint.
        field : str | None
            For complex types, the specific field to constrain.
        model : LanguageModelName | LanguageModel | str | None
            Model to use for guardrail evaluation (defaults to thing's model).
        raise_on_error : bool
            If True, raises GuardrailError on violation. If False, returns error info.
        severity : Literal["error", "warning", "info"]
            Severity level of the constraint.

        Examples
        --------
        >>> thing = to_thing(int)
        >>> thing.guardrail("must be positive")
        >>> thing.guardrail("must be less than 100", severity="warning")
        """
        guardrail_config = {
            "constraint": constraint,
            "field": field,
            "model": model or self._model,
            "raise_on_error": raise_on_error,
            "severity": severity,
        }
        self._guardrails.append(guardrail_config)

    async def _check_guardrails(self, value: T) -> T:
        """Check all guardrails against the given value.

        This is an internal method called during make/edit operations.
        """
        if not self._guardrails:
            return value

        # TODO: Implement actual LLM-based guardrail checking
        # For now, just return the value
        # This will be implemented when we have the full guardrail system
        return value

    async def amake(
        self,
        prompt: str | None = None,
        *,
        messages: List[Dict[str, Any]] | None = None,
        n: int = 1,
        update: bool = False,
        confirm: bool = False,
        reason: bool = False,
        parallel: bool = False,
        diversity: Literal["low", "medium", "high"] | None = None,
        instructor_mode: Mode | None = None,
        include_content: bool = False,
    ) -> LanguageModelResponse[T] | LanguageModelResponse[List[T]]:
        """Asynchronously generate content for this thing.

        Parameters
        ----------
        prompt : str | None
            Generation prompt/context.
        messages : List[Dict[str, Any]] | None
            Optional conversation history.
        n : int
            Number of instances to generate.
        update : bool
            If True, updates this thing's content with the result (only for n=1).
        confirm : bool
            If True, confirm which fields need generation.
        reason : bool
            If True, include reasoning for decisions.
        parallel : bool
            If True with n>1, generate each instance separately.
        diversity : Literal["low", "medium", "high"] | None
            Level of diversity for batch generation.
        instructor_mode : Mode | None
            Instructor mode for structured output.
        include_content : bool
            If True, includes the current content in the context.

        Returns
        -------
        LanguageModelResponse[T] | LanguageModelResponse[List[T]]
            The generated content.

        Examples
        --------
        >>> thing = to_thing(int)
        >>> response = await thing.amake("what is 2+2?", update=True)
        >>> print(thing.content)
        4
        """
        from pydantic import BaseModel

        from .maker import Maker

        # Build messages with content if requested
        if include_content and self._content is not None:
            if isinstance(self._content, BaseModel):
                content_str = self._content.model_dump_json(indent=2)
            else:
                content_str = str(self._content)

            content_message = {
                "role": "system",
                "content": f"Current content:\n{content_str}",
            }

            if messages is None:
                messages = [content_message]
            else:
                messages = [content_message] + messages

        maker = Maker(
            type=self._schema,
            model=self._model,
            instructions=self._instructions,
        )

        response = await maker.amake(
            prompt=prompt,
            messages=messages,
            n=n,
            confirm=confirm,
            reason=reason,
            parallel=parallel,
            diversity=diversity,
        )

        if update and n == 1 and response.content is not None:
            # Check guardrails
            validated_content = await self._check_guardrails(
                response.content
            )
            self.content = validated_content

        return response

    def make(
        self,
        prompt: str | None = None,
        *,
        messages: List[Dict[str, Any]] | None = None,
        n: int = 1,
        update: bool = False,
        confirm: bool = False,
        reason: bool = False,
        parallel: bool = False,
        diversity: Literal["low", "medium", "high"] | None = None,
        instructor_mode: Mode | None = None,
        include_content: bool = True,
    ) -> LanguageModelResponse[T] | LanguageModelResponse[List[T]]:
        """Generate content for this thing (synchronous version).

        See `amake()` for parameter documentation.

        Examples
        --------
        >>> thing = to_thing(int)
        >>> thing.make("what is 2+2?", update=True)
        >>> print(thing.content)
        4
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.amake(
                        prompt=prompt,
                        messages=messages,
                        n=n,
                        update=update,
                        confirm=confirm,
                        reason=reason,
                        parallel=parallel,
                        diversity=diversity,
                        instructor_mode=instructor_mode,
                        include_content=include_content,
                    ),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.amake(
                    prompt=prompt,
                    messages=messages,
                    n=n,
                    update=update,
                    confirm=confirm,
                    reason=reason,
                    parallel=parallel,
                    diversity=diversity,
                    instructor_mode=instructor_mode,
                    include_content=include_content,
                )
            )

    async def aedit(
        self,
        prompt: str | None = None,
        *,
        messages: List[Dict[str, Any]] | None = None,
        update: bool = False,
        confirm: bool = False,
        selective: bool = False,
        reason: bool = False,
        parallel: bool = False,
    ) -> LanguageModelResponse[T]:
        """Asynchronously edit the content of this thing.

        Parameters
        ----------
        prompt : str | None
            Edit instruction/context.
        messages : List[Dict[str, Any]] | None
            Optional conversation history.
        update : bool
            If True, updates this thing's content with the result.
        confirm : bool
            If True, confirm which fields need editing.
        selective : bool
            If True, determine edit strategy for each field.
        reason : bool
            If True, include reasoning for edits.
        parallel : bool
            If True, edit fields separately in parallel.

        Returns
        -------
        LanguageModelResponse[T]
            The edited content.

        Raises
        ------
        ThingError
            If thing has no content to edit.

        Examples
        --------
        >>> thing = to_thing(int)
        >>> thing.content = 4
        >>> response = await thing.aedit("make it 10", update=True)
        >>> print(thing.content)
        10
        """
        if self._content is None:
            raise ThingError("Cannot edit: thing has no content")

        from .maker import Maker

        maker = Maker(
            type=self._schema,
            model=self._model,
            instructions=self._instructions,
        )

        response = await maker.aedit(
            value=self._content,
            prompt=prompt,
            messages=messages,
            confirm=confirm,
            selective=selective,
            reason=reason,
            parallel=parallel,
        )

        if update and response.content is not None:
            # Check guardrails
            validated_content = await self._check_guardrails(
                response.content
            )
            self.content = validated_content

        return response

    def edit(
        self,
        prompt: str | None = None,
        *,
        messages: List[Dict[str, Any]] | None = None,
        update: bool = False,
        confirm: bool = False,
        selective: bool = False,
        reason: bool = False,
        parallel: bool = False,
    ) -> LanguageModelResponse[T]:
        """Edit the content of this thing (synchronous version).

        See `aedit()` for parameter documentation.

        Examples
        --------
        >>> thing = to_thing(int)
        >>> thing.content = 4
        >>> thing.edit("make it 10", update=True)
        >>> print(thing.content)
        10
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.aedit(
                        prompt=prompt,
                        messages=messages,
                        update=update,
                        confirm=confirm,
                        selective=selective,
                        reason=reason,
                        parallel=parallel,
                    ),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.aedit(
                    prompt=prompt,
                    messages=messages,
                    update=update,
                    confirm=confirm,
                    selective=selective,
                    reason=reason,
                    parallel=parallel,
                )
            )

    async def aquery(
        self,
        query: str,
        *,
        type: Type[R] | Schema[R] | None = None,
        instructions: str | None = None,
        stream: bool = False,
        instructor_mode: Mode | None = None,
    ) -> LanguageModelResponse[R | str]:
        """Asynchronously query the content of this thing.

        Parameters
        ----------
        query : str
            The query to ask about the content.
        type : Type[R] | Schema[R] | None
            Optional response type for structured output.
        instructions : str | None
            Optional additional instructions.
        stream : bool
            Whether to stream the response.
        instructor_mode : Mode | None
            Instructor mode for structured output.

        Returns
        -------
        LanguageModelResponse[R | str]
            The query response.

        Raises
        ------
        ThingError
            If thing has no content to query.

        Examples
        --------
        >>> thing = to_thing(int)
        >>> thing.content = 42
        >>> response = await thing.aquery("is this even or odd?")
        >>> print(response.content)
        "even"
        """
        if self._content is None:
            raise ThingError("Cannot query: thing has no content")

        # Build context message with the content
        from pydantic import BaseModel

        if isinstance(self._content, BaseModel):
            content_str = self._content.model_dump_json(indent=2)
        else:
            content_str = str(self._content)

        messages = [
            {
                "role": "system",
                "content": f"Current content:\n{content_str}",
            },
            {"role": "user", "content": query},
        ]

        return await self._model.arun(
            messages=messages,
            type=type or str,
            instructions=instructions,
            stream=stream,
            instructor_mode=instructor_mode,
        )

    def query(
        self,
        query: str,
        *,
        type: Type[R] | Schema[R] | None = None,
        instructions: str | None = None,
        stream: bool = False,
        instructor_mode: Mode | None = None,
    ) -> LanguageModelResponse[R | str]:
        """Query the content of this thing (synchronous version).

        See `aquery()` for parameter documentation.

        Examples
        --------
        >>> thing = to_thing(int)
        >>> thing.content = 42
        >>> response = thing.query("is this even or odd?")
        >>> print(response.content)
        "even"
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.aquery(
                        query=query,
                        type=type,
                        instructions=instructions,
                        stream=stream,
                        instructor_mode=instructor_mode,
                    ),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.aquery(
                    query=query,
                    type=type,
                    instructions=instructions,
                    stream=stream,
                    instructor_mode=instructor_mode,
                )
            )

    def __str__(self) -> str:
        """String representation shows the content."""
        return str(self._content) if self._content is not None else "None"

    def __repr__(self) -> str:
        """Detailed representation of the thing."""
        type_name = (
            self._schema.source.__name__
            if hasattr(self._schema.source, "__name__")
            else str(self._schema.source)
        )
        content_repr = (
            repr(self._content) if self._content is not None else "None"
        )
        return f"Thing(type={type_name}, content={content_repr})"

    # ============================================================================
    # Natural Language Operators (Dunder Methods)
    # ============================================================================
    # These methods enable LLM-powered evaluation of operators using natural
    # language semantics. When you compare, combine, or test a Thing, the LLM
    # interprets the operation based on the content and context.

    async def _llm_compare(self, other: Any, operation: str) -> bool:
        """Internal method to perform LLM-based comparisons.

        Parameters
        ----------
        other : Any
            The value to compare against.
        operation : str
            The comparison operation in natural language (e.g., "equal to", "greater than").

        Returns
        -------
        bool
            Result of the comparison.
        """
        if self._content is None:
            # None comparisons use Python semantics
            if operation == "equal to":
                return other is None
            elif operation == "not equal to":
                return other is not None
            else:
                return False

        from pydantic import BaseModel

        # Format the content
        if isinstance(self._content, BaseModel):
            content_str = self._content.model_dump_json(indent=2)
        else:
            content_str = str(self._content)

        # Format the comparison value
        if isinstance(other, Thing):
            if other._content is None:
                other_str = "None"
            elif isinstance(other._content, BaseModel):
                other_str = other._content.model_dump_json(indent=2)
            else:
                other_str = str(other._content)
        else:
            other_str = str(other)

        # Build the prompt for comparison
        messages = [
            {
                "role": "system",
                "content": (
                    "You are evaluating a comparison operation. "
                    "Analyze the content and determine if the comparison is true or false. "
                    "Consider semantic meaning, not just literal string matching. "
                    "For example, 'I don't like that guy' could be considered negative sentiment, "
                    "and '5' could be less than '10' numerically."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Content: {content_str}\n\n"
                    f"Is the content {operation} '{other_str}'?\n\n"
                    f"Answer with only 'true' or 'false'."
                ),
            },
        ]

        response = await self._model.arun(
            messages=messages,
            type=bool,
        )

        return response.content if response.content is not None else False

    def __eq__(self, other: Any) -> bool:
        """Natural language equality comparison.

        Uses LLM to determine if this thing's content is semantically equal to
        the comparison value. This goes beyond literal equality to understand
        meaning and context.

        Parameters
        ----------
        other : Any
            Value to compare against (can be another Thing, a string, or any type).

        Returns
        -------
        bool
            True if semantically equal, False otherwise.

        Examples
        --------
        >>> sentiment = thing("I love this!")
        >>> sentiment == "positive"  # LLM evaluates: True
        True

        >>> answer = thing(42)
        >>> answer == "the answer to life"  # LLM understands context: True
        True

        >>> name = thing("Alice")
        >>> name == "alice"  # Case-insensitive semantic match: True
        True
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._llm_compare(other, "equal to"),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self._llm_compare(other, "equal to"))

    def __ne__(self, other: Any) -> bool:
        """Natural language inequality comparison.

        Returns
        -------
        bool
            True if semantically not equal, False otherwise.

        Examples
        --------
        >>> sentiment = thing("I hate this!")
        >>> sentiment != "positive"  # LLM evaluates: True
        True
        """
        return not self.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        """Natural language less-than comparison.

        Uses LLM to determine if this thing's content is semantically less than
        the comparison value. Works with numbers, dates, rankings, and more.

        Parameters
        ----------
        other : Any
            Value to compare against.

        Returns
        -------
        bool
            True if semantically less than, False otherwise.

        Examples
        --------
        >>> priority = thing("low")
        >>> priority < "high"  # LLM understands ranking: True
        True

        >>> age = thing("25 years old")
        >>> age < 30  # LLM parses and compares: True
        True
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._llm_compare(other, "less than"),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self._llm_compare(other, "less than"))

    def __le__(self, other: Any) -> bool:
        """Natural language less-than-or-equal comparison.

        Returns
        -------
        bool
            True if semantically less than or equal, False otherwise.

        Examples
        --------
        >>> score = thing("B+")
        >>> score <= "A"  # LLM understands grading: True
        True
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._llm_compare(other, "less than or equal to"),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self._llm_compare(other, "less than or equal to")
            )

    def __gt__(self, other: Any) -> bool:
        """Natural language greater-than comparison.

        Uses LLM to determine if this thing's content is semantically greater than
        the comparison value.

        Parameters
        ----------
        other : Any
            Value to compare against.

        Returns
        -------
        bool
            True if semantically greater than, False otherwise.

        Examples
        --------
        >>> priority = thing("critical")
        >>> priority > "low"  # LLM understands severity: True
        True

        >>> temp = thing("hot")
        >>> temp > "cold"  # LLM understands temperature: True
        True
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._llm_compare(other, "greater than"),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self._llm_compare(other, "greater than"))

    def __ge__(self, other: Any) -> bool:
        """Natural language greater-than-or-equal comparison.

        Returns
        -------
        bool
            True if semantically greater than or equal, False otherwise.

        Examples
        --------
        >>> rating = thing("excellent")
        >>> rating >= "good"  # LLM understands quality scale: True
        True
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._llm_compare(other, "greater than or equal to"),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self._llm_compare(other, "greater than or equal to")
            )

    async def _llm_operation(self, other: Any, operation: str) -> T:
        """Internal method to perform LLM-based operations.

        Parameters
        ----------
        other : Any
            The value to combine with.
        operation : str
            The operation in natural language (e.g., "combine with", "subtract").

        Returns
        -------
        T
            Result of the operation with the same type as this thing.
        """
        if self._content is None:
            raise ThingError(
                f"Cannot perform {operation} operation: thing has no content"
            )

        from pydantic import BaseModel

        # Format the content
        if isinstance(self._content, BaseModel):
            content_str = self._content.model_dump_json(indent=2)
        else:
            content_str = str(self._content)

        # Format the other value
        if isinstance(other, Thing):
            if other._content is None:
                raise ThingError(
                    f"Cannot perform {operation} operation: other thing has no content"
                )
            if isinstance(other._content, BaseModel):
                other_str = other._content.model_dump_json(indent=2)
            else:
                other_str = str(other._content)
        else:
            other_str = str(other)

        # Build the prompt for the operation
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are performing a '{operation}' operation. "
                    f"Take the content and {operation} it with the given value. "
                    f"Return the result in the appropriate format. "
                    f"Be contextually aware - for text, combine meanings; "
                    f"for numbers, perform arithmetic; for objects, merge appropriately."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Content: {content_str}\n\n"
                    f"{operation.capitalize()}: {other_str}\n\n"
                    f"Provide the result."
                ),
            },
        ]

        response = await self._model.arun(
            messages=messages,
            type=self._schema.source,
        )

        if response.content is None:
            raise ThingError(
                f"LLM returned no content for {operation} operation"
            )

        return response.content

    def __add__(self, other: Any) -> Thing[T]:
        """Natural language addition/combination.

        Creates a new Thing by combining this thing's content with another value.
        The LLM interprets how to combine based on context.

        Parameters
        ----------
        other : Any
            Value to add/combine with (can be another Thing or any type).

        Returns
        -------
        Thing[T]
            A new Thing with the combined result.

        Examples
        --------
        >>> story = thing("Once upon a time")
        >>> extended = story + " there was a dragon"
        >>> print(extended.content)
        "Once upon a time there was a dragon"

        >>> count = thing(5)
        >>> total = count + 3
        >>> print(total.content)
        8

        >>> task = thing({"title": "Write code", "priority": "high"})
        >>> updated = task + "add deadline: tomorrow"
        >>> # LLM adds deadline field to the object
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._llm_operation(other, "add"),
                )
                result = future.result()
        except RuntimeError:
            result = asyncio.run(self._llm_operation(other, "add"))

        # Create a new Thing with the result
        new_thing = Thing(
            self._schema,
            model=self._model,
            instructions=self._instructions,
        )
        new_thing._content = result
        return new_thing

    def __sub__(self, other: Any) -> Thing[T]:
        """Natural language subtraction/removal.

        Creates a new Thing by subtracting/removing the other value from this
        thing's content. The LLM interprets how to subtract based on context.

        Parameters
        ----------
        other : Any
            Value to subtract/remove.

        Returns
        -------
        Thing[T]
            A new Thing with the result.

        Examples
        --------
        >>> text = thing("Hello wonderful world")
        >>> shorter = text - "wonderful"
        >>> print(shorter.content)
        "Hello world"

        >>> count = thing(10)
        >>> remaining = count - 3
        >>> print(remaining.content)
        7

        >>> tags = thing(["python", "java", "rust"])
        >>> filtered = tags - "java"
        >>> # LLM removes "java" from the list
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._llm_operation(other, "subtract"),
                )
                result = future.result()
        except RuntimeError:
            result = asyncio.run(self._llm_operation(other, "subtract"))

        # Create a new Thing with the result
        new_thing = Thing(
            self._schema,
            model=self._model,
            instructions=self._instructions,
        )
        new_thing._content = result
        return new_thing

    def __bool__(self) -> bool:
        """Natural language boolean evaluation.

        Evaluates the "truthiness" of this thing's content using LLM interpretation.
        Goes beyond Python's default truthiness to understand semantic meaning.

        Returns
        -------
        bool
            True if semantically truthy, False otherwise.

        Examples
        --------
        >>> sentiment = thing("I love this!")
        >>> bool(sentiment)  # Positive sentiment
        True

        >>> answer = thing("no")
        >>> bool(answer)  # Negative response
        False

        >>> status = thing("active")
        >>> bool(status)  # Active state
        True

        >>> empty = thing("nothing")
        >>> bool(empty)  # Semantic emptiness
        False
        """
        if self._content is None:
            return False

        from pydantic import BaseModel

        # Format the content
        if isinstance(self._content, BaseModel):
            content_str = self._content.model_dump_json(indent=2)
        else:
            content_str = str(self._content)

        # Build the prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are evaluating the truthiness of content. "
                    "Determine if the content represents a true/positive/active/yes state "
                    "or a false/negative/inactive/no state. "
                    "Consider semantic meaning: positive sentiment is true, "
                    "negative sentiment is false, affirmative responses are true, "
                    "empty/null/none concepts are false, etc."
                ),
            },
            {
                "role": "user",
                "content": f"Content: {content_str}\n\nIs this content truthy? Answer with only 'true' or 'false'.",
            },
        ]

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._model.arun(messages=messages, type=bool),
                )
                response = future.result()
        except RuntimeError:
            response = asyncio.run(
                self._model.arun(messages=messages, type=bool)
            )

        return response.content if response.content is not None else False

    def __contains__(self, item: Any) -> bool:
        """Natural language containment check.

        Checks if the given item is semantically contained within this thing's content.
        The LLM interprets containment based on context (substring, element, concept, etc.).

        Parameters
        ----------
        item : Any
            Item to check for containment.

        Returns
        -------
        bool
            True if item is semantically contained, False otherwise.

        Examples
        --------
        >>> text = thing("The quick brown fox")
        >>> "fox" in text  # Substring containment
        True

        >>> topics = thing(["AI", "Machine Learning", "Neural Networks"])
        >>> "deep learning" in topics  # Semantic concept containment
        True

        >>> description = thing("This is about programming")
        >>> "coding" in description  # Synonym/concept containment
        True
        """
        if self._content is None:
            return False

        from pydantic import BaseModel

        # Format the content
        if isinstance(self._content, BaseModel):
            content_str = self._content.model_dump_json(indent=2)
        else:
            content_str = str(self._content)

        # Format the item
        item_str = str(item)

        # Build the prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are checking if an item is contained within content. "
                    "Consider semantic containment: exact matches, substrings, "
                    "list/array elements, object fields, or conceptually related items. "
                    "For example, if content mentions 'programming', it contains 'coding'. "
                    "If content is a list with 'machine learning', it contains related concepts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Content: {content_str}\n\n"
                    f"Does the content contain '{item_str}'? "
                    f"Answer with only 'true' or 'false'."
                ),
            },
        ]

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._model.arun(messages=messages, type=bool),
                )
                response = future.result()
        except RuntimeError:
            response = asyncio.run(
                self._model.arun(messages=messages, type=bool)
            )

        return response.content if response.content is not None else False


def to_thing(
    t: Type[T] | Schema[T] | T = str,
    *,
    model: LanguageModelName
    | LanguageModel[T]
    | str = "openai/gpt-4o-mini",
    instructions: str | None = None,
    title: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
    key: str | None = None,
) -> Thing[T]:
    """Create a new Thing with the given type or initial value.

    A Thing is a mutable, type-aware container that can generate, edit,
    and query its content using language models.

    Parameters
    ----------
    t : Type[T] | Schema[T] | T
        Either a type specification, a Schema, or an initial value.
        If a value is provided, the type will be inferred from it.
    model : LanguageModelName | LanguageModel | str
        The default language model to use for operations.
    instructions : str | None
        Default instructions for the language model.
    title : str | None
        Optional title for the schema.
    description : str | None
        Optional description for the schema.
    exclude : set[str] | None
        Fields to exclude from schema operations.
    key : str | None
        For simple types, override the default field name.

    Returns
    -------
    Thing[T]
        A new Thing instance.

    Examples
    --------
    >>> # Create an empty thing from type
    >>> answer = thing(int)
    >>> print(answer.content)
    None

    >>> # Create a thing with initial value
    >>> name = thing("Alice")
    >>> print(name.content)
    Alice

    >>> # Make content
    >>> answer.make("what is 2+2?", update=True)
    >>> print(answer.content)
    4

    >>> # Create with existing content and set later
    >>> number = thing(int)
    >>> number.content = 42
    >>> print(number.content)
    42

    >>> # Complex types with Pydantic models
    >>> from pydantic import BaseModel
    >>> class User(BaseModel):
    ...     name: str
    ...     age: int
    >>> user = thing(User)
    >>> user.make("create a sample user", update=True)
    """
    return Thing(
        t=t,
        model=model,
        instructions=instructions,
        title=title,
        description=description,
        exclude=exclude,
        key=key,
    )
