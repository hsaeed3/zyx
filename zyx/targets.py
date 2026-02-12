"""zyx.targets"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Generic,
    Type,
    TypeAlias,
    TypeVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .operations.validate import ValidationResult
    from ._aliases import (
        PydanticAIInstructions,
        PydanticAIModelSettings,
        PydanticAIUsageLimits,
    )
    from ._types import SourceParam, ModelParam, ContextType, ToolType


Deps = TypeVar("Deps")
Output = TypeVar("Output")


TargetHook: TypeAlias = Literal["complete", "error"]


@dataclass
class Target(Generic[Output]):
    """
    A structured way to define the `target` or output to generate
    through a semantic operation.
    """

    target: Output | Type[Output]
    """The target type or value that will be generated/mutated by
    semantic operations."""

    name: str | None = field(default=None)
    """An optional human readable name for this target type, this is useful
    for targets of primitive types as well as for better model understanding
    of a target."""

    description: str | None = field(default=None)
    """An optional human readable description of this target type."""

    instructions: str | None = field(default=None)
    """Specific instructions for the model to follow when generating this target."""

    constraints: List[str] | None = field(default=None)
    """Optional list of constraints the parsed value must satisfy (used by parse/aparse)."""

    model: ModelParam | None = field(default=None)
    """Default model/pydantic ai agent to use for operations on this target (e.g. ``validate``)."""

    constraints: str | List[str] | None = field(default=None)
    """An optional list of constraints that this target type must adhere to,
    these can be explicitly validated by a model if needed."""

    _field_hooks: Dict[str, List[tuple[Callable[[Any], Any], bool, bool]]] = (
        field(init=False, default_factory=dict)
    )
    """A dictionary of field hooks that can be used to validate/mutate this target
    as it is being generated/mutated.

    You can set a field hook on a target by using the `@on_field` decorator."""

    _prebuilt_hooks: Dict[
        TargetHook, List[tuple[Callable[[Any], Any], bool, bool]]
    ] = field(init=False, default_factory=dict)
    """A list of prebuilt hooks that can be used to validate/mutate this target
    based on a certain event or condition.

    You can apply a prebuilt hook on a target by using the `@on` decorator."""

    def on_field(
        self,
        field: str | None = None,
        retry: bool = True,
        update: bool = False,
    ):
        """
        Decorator to register a 'field hook' on this target. A field hook
        is a function that can be used to validate/mutate a specific field
        on the event it is generated or mutated by a model.

        For targets with types such as `str`, `int`, with no field names, you
        can leave the `field` argument as None.
        """

        def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
            key = field or "__self__"
            self._field_hooks.setdefault(key, []).append((fn, retry, update))
            return fn

        return decorator

    def on(
        self,
        event: TargetHook,
        retry: bool = True,
        update: bool = False,
    ):
        """
        Decorator to register a 'prebuilt hook' on this target.

        Hooks include:
        - `complete`: Called when the target is generated/mutated successfully.
        - `error`: Called when the target is generated/mutated with an error.
        """

        def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
            self._prebuilt_hooks.setdefault(event, []).append(
                (fn, retry, update)
            )
            return fn

        return decorator

    def validate(
        self,
        source: SourceParam,
        *,
        context: ContextType | List[ContextType] | None = None,
        constraints: List[str] | None = None,
        raise_on_error: bool = True,
        model: ModelParam | None = None,
        model_settings: PydanticAIModelSettings | None = None,
        instructions: PydanticAIInstructions | None = None,
        tools: ToolType | List[ToolType] | None = None,
        deps: Deps | None = None,
        usage_limits: PydanticAIUsageLimits | None = None,
    ) -> Output | ValidationResult[Output]:
        """
        Validate a given input `source` against the constraints (or any additional
        constraints) defined on this target, using a `pydantic_ai` model or agent.

        Args:
            source : SourceParam
                The source value to validate against the constraints.
            context : ContextType | List[ContextType] | None = None
                Additional context or conversation history to use for the validation.
            constraints : List[str] | None = None
                Optional list of constraints the parsed value must satisfy (used by parse/aparse).
            raise_on_error : bool = True
                If `True`, raise an error if the constraints are not satisfied.
            model : ModelParam | None = None
                The model or agent to use for the validation.
            model_settings : PydanticAIModelSettings | None = None
                Model settings to use for the validation.
            instructions : PydanticAIInstructions | None = None
                Specific instructions for the model to follow when validating this target.
            tools : ToolType | List[ToolType] | None = None
                Tools to use for the validation.
            deps : Deps | None = None

        Returns:
            Output | ValidationResult[Output]
                The validated output or a `ValidationResult` object if `raise_on_error` is `False`.
        """
        from .operations.validate import validate, ValidationResult

        if not model and not self.model:
            raise ValueError(
                "No model/default model provided for validation. Please either set the `model` attribute on"
                "this target or provide a model to the `validate` method."
            )
        _model = model if model is not None else self.model

        output = validate(
            source=source,
            target=self,
            context=context,
            constraints=constraints,
            raise_on_error=raise_on_error,
            model=_model,  # type: ignore[arg-type]
            model_settings=model_settings,
            instructions=instructions,
            tools=tools,
            deps=deps,
            usage_limits=usage_limits,
        )
        if isinstance(output, ValidationResult):
            return output  # type: ignore[return-value]
        return output.output  # type: ignore[return-value]

    async def avalidate(
        self,
        source: SourceParam,
        *,
        context: ContextType | List[ContextType] | None = None,
        constraints: List[str] | None = None,
        raise_on_error: bool = True,
        model: ModelParam | None = None,
        model_settings: PydanticAIModelSettings | None = None,
        instructions: PydanticAIInstructions | None = None,
        tools: ToolType | List[ToolType] | None = None,
        deps: Deps | None = None,
        usage_limits: PydanticAIUsageLimits | None = None,
    ) -> Output | ValidationResult[Output]:
        """
        Validate a given input `source` against the constraints (or any additional
        constraints) defined on this target, using a `pydantic_ai` model or agent.

        Args:
            source : SourceParam
                The source value to validate against the constraints.
            context : ContextType | List[ContextType] | None = None
                Additional context or conversation history to use for the validation.
            constraints : List[str] | None = None
                Optional list of constraints the parsed value must satisfy (used by parse/aparse).
            raise_on_error : bool = True
                If `True`, raise an error if the constraints are not satisfied.
            model : ModelParam | None = None
                The model or agent to use for the validation.
            model_settings : PydanticAIModelSettings | None = None
                Model settings to use for the validation.
            instructions : PydanticAIInstructions | None = None
                Specific instructions for the model to follow when validating this target.
            tools : ToolType | List[ToolType] | None = None
                Tools to use for the validation.
            deps : Deps | None = None

        Returns:
            Output | ValidationResult[Output]
                The validated output or a `ValidationResult` object if `raise_on_error` is `False`.
        """
        from .operations.validate import avalidate, ValidationResult

        if not model and not self.model:
            raise ValueError(
                "No model/default model provided for validation. Please either set the `model` attribute on"
                "this target or provide a model to the `validate` method."
            )
        _model = model if model is not None else self.model

        output = await avalidate(
            source=source,
            target=self,
            context=context,
            constraints=constraints,
            raise_on_error=raise_on_error,
            model=_model,  # type: ignore[arg-type]
            model_settings=model_settings,
            instructions=instructions,
            tools=tools,
            deps=deps,
            usage_limits=usage_limits,
        )
        if isinstance(output, ValidationResult):
            return output  # type: ignore[return-value]
        return output.output  # type: ignore[return-value]

    def __call__(
        self,
        source: SourceParam,
        *,
        context: ContextType | List[ContextType] | None = None,
        constraints: List[str] | None = None,
        raise_on_error: bool = True,
        model: ModelParam | None = None,
        model_settings: PydanticAIModelSettings | None = None,
        instructions: PydanticAIInstructions | None = None,
        tools: ToolType | List[ToolType] | None = None,
        deps: Deps | None = None,
        usage_limits: PydanticAIUsageLimits | None = None,
    ) -> Output | ValidationResult[Output]:
        """
        Validate a given input `source` against the constraints (or any additional
        constraints) defined on this target, using a `pydantic_ai` model or agent.

        Args:
            source : SourceParam
                The source value to validate against the constraints.
            context : ContextType | List[ContextType] | None = None
                Additional context or conversation history to use for the validation.
            constraints : List[str] | None = None
                Optional list of constraints the parsed value must satisfy (used by parse/aparse).
            raise_on_error : bool = True
                If `True`, raise an error if the constraints are not satisfied.
            model : ModelParam | None = None
                The model or agent to use for the validation.
            model_settings : PydanticAIModelSettings | None = None
                Model settings to use for the validation.
            instructions : PydanticAIInstructions | None = None
                Specific instructions for the model to follow when validating this target.
            tools : ToolType | List[ToolType] | None = None
                Tools to use for the validation.
            deps : Deps | None = None
                Additional dependencies to use for the validation.

        Returns:
            Output | ValidationResult[Output]
                The validated output or a `ValidationResult` object if `raise_on_error` is `False`.
        """
        return self.validate(
            source=source,
            context=context,
            constraints=constraints,
            raise_on_error=raise_on_error,
            model=model,
            model_settings=model_settings,
            instructions=instructions,
            tools=tools,
            deps=deps,
            usage_limits=usage_limits,
        )


def target(
    target: Output | Type[Output],
    name: str | None = None,
    description: str | None = None,
    instructions: str | None = None,
    constraints: List[str] | None = None,
    model: ModelParam | None = None,
) -> Target[Output]:
    """
    A decorator/function to create a new `Target` object from a
    given type or value.

    Args:
        target: Output | Type[Output]
            The target type or value to create a `Target` object from.
        name: str | None = None
            An optional human readable name for this target type, this is useful
            for targets of primitive types as well as for better model understanding
            of a target.
        description: str | None = None
            An optional human readable description of this target type.
        instructions: str | None = None
            Specific instructions for the model to follow when generating this target.
        constraints: List[str] | None = None
            Optional list of constraints the parsed value must satisfy (used by parse/aparse).
        model: ModelParam | None = None
            Default model/pydantic ai agent to use for operations on this target (e.g. ``validate``).

    Returns:
        Target[Output]
            A new `Target` object from the given type or value.
    """
    return Target(
        target=target,
        name=name,
        description=description,
        instructions=instructions,
        constraints=constraints,
        model=model,
    )
