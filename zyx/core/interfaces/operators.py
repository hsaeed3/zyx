"""zyx.core.interfaces.operators"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from ..models.language.model import LanguageModel

__all__ = ("Operators",)


T = TypeVar("T")


class Operators(ABC, Generic[T]):
    """Mixin class providing LLM-powered dunder methods for natural language operations.
    
    This mixin enables Python operators (==, <, +, -, bool, in) to work with
    natural language semantics using language models. Classes that inherit from
    this mixin can compare, combine, and test their content using LLM interpretation.
    
    Required Attributes (must be implemented by subclass)
    ------------------------------------------------------
    _model : LanguageModel
        The language model to use for LLM operations.
    _content : T | None
        The content to operate on.
    _schema : Schema[T] (for operations that need type info)
        The schema of the content type.
    
    Abstract Methods
    ----------------
    _get_content_for_llm() -> str | None
        Returns the content formatted as a string for LLM consumption.
    _get_model_for_llm() -> LanguageModel
        Returns the language model to use for operations.
    _create_new_instance(content: T) -> Any
        Creates a new instance of the class with the given content.
        Used by operations that return new instances (like __add__).
    """

    @abstractmethod
    def _get_content_for_llm(self) -> str | None:
        """Get the content formatted for LLM operations.
        
        Returns
        -------
        str | None
            The content as a string, or None if no content.
        """
        pass

    @abstractmethod
    def _get_model_for_llm(self) -> LanguageModel:
        """Get the language model for LLM operations.
        
        Returns
        -------
        LanguageModel
            The language model to use.
        """
        pass

    # ============================================================================
    # Natural Language Operators (Dunder Methods)
    # ============================================================================
    # These methods provide comparison and query operations that work for both
    # mutable (Thing) and immutable (Bit) content types.

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
        content_str = self._get_content_for_llm()
        
        if content_str is None:
            # None comparisons use Python semantics
            if operation == "equal to":
                return other is None
            elif operation == "not equal to":
                return other is not None
            else:
                return False

        # Format the comparison value
        if hasattr(other, '_get_content_for_llm'):
            other_str = other._get_content_for_llm()
            if other_str is None:
                other_str = "None"
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

        model = self._get_model_for_llm()
        response = await model.arun(
            messages=messages,
            type=bool,
        )

        return response.content if response.content is not None else False

    def __eq__(self, other: Any) -> bool:
        """Natural language equality comparison.

        Uses LLM to determine if this content is semantically equal to
        the comparison value. This goes beyond literal equality to understand
        meaning and context.

        Parameters
        ----------
        other : Any
            Value to compare against.

        Returns
        -------
        bool
            True if semantically equal, False otherwise.

        Examples
        --------
        >>> bit = to_bit("I love this!")
        >>> bit == "positive"  # LLM evaluates: True
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
        """
        return not self.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        """Natural language less-than comparison.

        Uses LLM to determine if this content is semantically less than
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
        >>> bit = to_bit("low priority")
        >>> bit < "high priority"  # LLM understands ranking: True
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

        Uses LLM to determine if this content is semantically greater than
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
        >>> bit = to_bit("critical priority")
        >>> bit > "low priority"  # LLM understands severity: True
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

    def __bool__(self) -> bool:
        """Natural language boolean evaluation.

        Evaluates the "truthiness" of this content using LLM interpretation.
        Goes beyond Python's default truthiness to understand semantic meaning.

        Returns
        -------
        bool
            True if semantically truthy, False otherwise.

        Examples
        --------
        >>> bit = to_bit("I love this!")
        >>> bool(bit)  # Positive sentiment
        True

        >>> bit = to_bit("no")
        >>> bool(bit)  # Negative response
        False
        """
        content_str = self._get_content_for_llm()
        
        if content_str is None:
            return False

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
                model = self._get_model_for_llm()
                future = executor.submit(
                    asyncio.run,
                    model.arun(messages=messages, type=bool),
                )
                response = future.result()
        except RuntimeError:
            model = self._get_model_for_llm()
            response = asyncio.run(
                model.arun(messages=messages, type=bool)
            )

        return response.content if response.content is not None else False

    def __contains__(self, item: Any) -> bool:
        """Natural language containment check.

        Checks if the given item is semantically contained within this content.
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
        >>> bit = to_bit("The quick brown fox")
        >>> "fox" in bit  # Substring containment
        True

        >>> bit = to_bit("This is about programming")
        >>> "coding" in bit  # Synonym/concept containment
        True
        """
        content_str = self._get_content_for_llm()
        
        if content_str is None:
            return False

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
                model = self._get_model_for_llm()
                future = executor.submit(
                    asyncio.run,
                    model.arun(messages=messages, type=bool),
                )
                response = future.result()
        except RuntimeError:
            model = self._get_model_for_llm()
            response = asyncio.run(
                model.arun(messages=messages, type=bool)
            )

        return response.content if response.content is not None else False

