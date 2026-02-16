"""zyx.operations.expressions"""

from __future__ import annotations

from typing import Any, List, TypeVar, TYPE_CHECKING

from .._aliases import (
    PydanticAIInstructions,
    PydanticAIModelSettings,
    PydanticAIUsageLimits,
)
from .._types import (
    ContextType,
    ModelParam,
    SourceParam,
    ToolType,
    AttachmentType,
)
from .parse import parse

if TYPE_CHECKING:
    from .._utils._observer import Observer

__all__ = ("expr", "Expressions")


Deps = TypeVar("Deps")
Output = TypeVar("Output")


class Expressions:
    """Helper class for semantic expression evaluation.

    Provides Pythonic expression methods (e.g., `==`, `in`, `bool()`) that
    evaluate semantic questions against a source or context using LLMs.
    """

    def __init__(
        self,
        source: SourceParam | None = None,
        context: ContextType | List[ContextType] | None = None,
        *,
        model: ModelParam = "openai:gpt-4o-mini",
        model_settings: PydanticAIModelSettings | None = None,
        attachments: AttachmentType | List[AttachmentType] | None = None,
        instructions: PydanticAIInstructions | None = None,
        tools: ToolType | List[ToolType] | None = None,
        deps: Deps | None = None,
        usage_limits: PydanticAIUsageLimits | None = None,
        observe: bool | Observer | None = None,
    ):
        """Initialize an Expressions instance.

        Args:
            source: The source value to evaluate expressions against.
            context: Optional context or conversation history.
            model: The model to use for evaluation.
            model_settings: Model settings (e.g., temperature).
            attachments: Attachments provided to the agent.
            instructions: Additional instructions/hints for the model.
            tools: List of tools available to the model.
            deps: Optional dependencies for this operation.
            usage_limits: Usage limits (token/request) configuration.
        """
        self._source = source
        self._context = context
        self._model = model
        self._model_settings = model_settings
        self._attachments = attachments
        self._instructions = instructions
        self._tools = tools
        self._deps = deps
        self._usage_limits = usage_limits
        self._observe = observe

    def _evaluate(
        self,
        question: str,
        output_type: type[Output] = bool,  # type: ignore[assignment]
    ) -> Output:
        """Internal method to evaluate a semantic question.

        Args:
            question: The semantic question to evaluate.
            output_type: The expected output type (default: bool).

        Returns:
            The evaluated result of the specified type.
        """
        # Combine source and context for the prompt
        prompt_parts: List[str] = []
        if self._source is not None:
            prompt_parts.append(f"Source:\n{self._source}")
        prompt_parts.append(f"Question: {question}")

        prompt = "\n\n".join(prompt_parts)

        # Combine context with the prompt
        combined_context: ContextType | List[ContextType] | None = None
        if self._context:
            if isinstance(self._context, list):
                # Type: ignore needed because prompt (str) is a valid ContextType
                combined_context = [*self._context, prompt]  # type: ignore[list-item]
            else:
                combined_context = [self._context, prompt]
        else:
            combined_context = prompt

        result = parse(
            source=self._source if self._source is not None else "",
            target=output_type,
            context=combined_context,
            model=self._model,
            model_settings=self._model_settings,
            attachments=self._attachments,
            instructions=self._instructions,
            tools=self._tools,
            deps=self._deps,
            usage_limits=self._usage_limits,
            observe=self._observe,
        )

        return result.output  # type: ignore[return-value]

    def __bool__(self) -> bool:
        """Evaluate whether the content has a positive/affirmative meaning.

        Returns:
            True if the content is semantically positive/affirmative, False otherwise.
        """
        return self._evaluate(
            "Is this statement true or does this content have a positive/affirmative meaning?",
            bool,
        )

    def __eq__(self, other: Any) -> bool:
        """Evaluate semantic equality with another value.

        Args:
            other: The value to compare against (must be a string for semantic comparison).

        Returns:
            True if the content semantically means or represents the other value.
        """
        if isinstance(other, str):
            return self._evaluate(
                f"Does this content semantically mean or represent '{other}'?",
                bool,
            )
        return False

    def __ne__(self, other: Any) -> bool:
        """Evaluate semantic inequality with another value.

        Args:
            other: The value to compare against.

        Returns:
            True if the content does not semantically mean or represent the other value.
        """
        return not self.__eq__(other)

    def __contains__(self, item: str) -> bool:
        """Evaluate whether the content contains or expresses a concept.

        Args:
            item: The concept or item to check for.

        Returns:
            True if the content contains or expresses the concept.
        """
        return self._evaluate(
            f"Does this content contain or express the concept of '{item}'?",
            bool,
        )


def expr(
    source: SourceParam | None = None,
    context: ContextType | List[ContextType] | None = None,
    *,
    model: ModelParam = "openai:gpt-4o-mini",
    model_settings: PydanticAIModelSettings | None = None,
    attachments: AttachmentType | List[AttachmentType] | None = None,
    instructions: PydanticAIInstructions | None = None,
    tools: ToolType | List[ToolType] | None = None,
    deps: Deps | None = None,
    usage_limits: PydanticAIUsageLimits | None = None,
    observe: bool | Observer | None = None,
) -> Expressions:
    """Create an Expressions instance for semantic expression evaluation.

    Args:
        source (SourceParam | None): The source value to evaluate expressions against. Defaults to None.
        context (ContextType | List[ContextType] | None): Optional context or conversation history. Defaults to None.
        model (ModelParam): The model to use for evaluation. Defaults to "openai:gpt-4o-mini".
        model_settings (PydanticAIModelSettings | None): Model settings (e.g., temperature). Defaults to None.
        attachments (AttachmentType | List[AttachmentType] | None): Attachments provided to the agent. Defaults to None.
        instructions (PydanticAIInstructions | None): Additional instructions/hints for the model. Defaults to None.
        tools (ToolType | List[ToolType] | None): List of tools available to the model. Defaults to None.
        deps (Deps | None): Optional dependencies for this operation. Defaults to None.
        usage_limits (PydanticAIUsageLimits | None): Usage limits (token/request) configuration. Defaults to None.

    Returns:
        Expressions: An Expressions instance that can be used for semantic expression evaluation.

    Example:
        ```python
        from zyx.operations.expressions import expr

        # Evaluate boolean expression
        result = bool(expr("The weather is sunny today"))

        # Evaluate semantic equality
        result = expr("It's a beautiful day") == "nice weather"

        # Check if content contains a concept
        result = "happiness" in expr("I feel great today")
        ```
    """
    return Expressions(
        source=source,
        context=context,
        model=model,
        model_settings=model_settings,
        attachments=attachments,
        instructions=instructions,
        tools=tools,
        deps=deps,
        usage_limits=usage_limits,
        observe=observe,
    )
