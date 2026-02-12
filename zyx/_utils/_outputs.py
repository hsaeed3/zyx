"""zyx.utils.outputs"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from typing import Any, Generic, List, Type, TypeVar

from pydantic import BaseModel, TypeAdapter

from .._aliases import PydanticAIAgentStream, PydanticAIAgentResult
from .._processing._outputs import (
    is_simple_type,
    normalize_output_target,
    partial_output_model,
)
from ..targets import Target


Output = TypeVar("Output")


@dataclasses.dataclass
class OutputBuilder(Generic[Output]):
    """Tracker/builder object that tracks the state of an output across the
    execution of a semantic operation's graph.

    This class provides the following functionality:
    - Type normalization of `target` types and values to a compatible type usable
    as the `output_type` parameter when running a Pydantic AI agent.
    - Creating partial output models for streaming/incremental validation.
    - Tracking output state across multiple agent runs.
    - Completeness validation and tracking.
    - Merging outputs from multiple agent runs.
    """

    target: Output | Type[Output] | Target[Output]
    """The source type or value that will be generated or mutated following the
    execution of a semantic operation."""

    _normalized: Type[Output | BaseModel] | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _partial_type: Type[BaseModel] | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _partial: Output | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _history: List[Output] = dataclasses.field(
        default_factory=list, init=False, repr=False
    )
    _filled_fields: set[str] = dataclasses.field(
        default_factory=set, init=False, repr=False
    )
    _is_simple: bool | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _is_value: bool | None = dataclasses.field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.target, Target):
            self.target = self.target.target
            target = self.target
        else:
            target = self.target

        self._is_value = not isinstance(target, type)

        self._normalized = normalize_output_target(target)
        self._is_simple = is_simple_type(self._normalized)

    @property
    def normalized(self) -> Type[Output | BaseModel]:
        """Normalized representation of the input `target`, compatible when used as
        the `output_type` parameter when running a Pydantic AI agent."""
        return self._normalized  # type: ignore

    @property
    def is_simple_type(self) -> bool:
        """Whether this normalized type is a 'simple'/primitive type.

        NOTE: Pydantic models are also considered simple types.
        """
        return self._is_simple  # type: ignore

    @property
    def is_value(self) -> bool:
        """Whether the target is a value rather than a type."""
        return self._is_value  # type: ignore

    @property
    def history(self) -> List[Output]:
        """All output states recorded."""
        return list(self._history)

    @property
    def partial_type(self) -> Type[BaseModel | Any]:
        """Optional representation of the normalized type used when streaming
        or incrementally validating output.

        For simple types that are not Pydantic models, this returns the
        type as-is.
        """
        if self._partial_type is not None:
            return self._partial_type

        if self._is_simple or not (
            isinstance(self._normalized, type)
            and issubclass(self._normalized, BaseModel)
        ):
            if hasattr(self._normalized, "model_fields"):
                pass
            self._partial_type = self._normalized  # type: ignore
            return self._partial_type  # type: ignore

        self._partial_type = partial_output_model(self._normalized)
        return self._partial_type

    @property
    def partial(self) -> Output | None:
        """The 'current' state/partial representation of the output. This property
        mutates as agent runs or streams are used to update this object."""
        return self._partial

    @property
    def field_count(self) -> int:
        """Returns the number of fields in the normalized type. For primitive/simple types without
        fields, this returns 0."""
        if self._is_simple:
            return 0
        if isinstance(self._normalized, type) and issubclass(
            self._normalized, BaseModel
        ):
            return len(self._normalized.model_fields)
        return 0

    @property
    def field_names(self) -> List[str]:
        """Returns the names of all fields in the normalized type. For primitive/simple types without
        fields, this returns an empty list."""
        if self._is_simple:
            return []
        if isinstance(self._normalized, type) and issubclass(
            self._normalized, BaseModel
        ):
            return list(self._normalized.model_fields.keys())
        return []

    @property
    def is_complete(self) -> bool:
        """Whether all content or fields of this output have been populated."""
        if self._partial is None:
            return False
        if self._is_simple:
            return True
        if isinstance(self._partial, BaseModel):
            for name in self.field_names:
                if getattr(self._partial, name, None) is None:
                    return False
            return True
        return True

    @property
    def missing_fields(self) -> List[str]:
        """Returns the names of all fields that are not yet populated."""
        if self._partial is None:
            return list(self.field_names)
        if self._is_simple:
            return []
        missing: List[str] = []
        if isinstance(self._partial, BaseModel):
            for name in self.field_names:
                if getattr(self._partial, name, None) is None:
                    missing.append(name)
        return missing

    def update(self, updates: Any) -> Output:
        """General update method for setting/updating the `current`/partial
        state of the target being composed by this builder.

        Args:
            updates: The new value to set. Can be:
                - A direct value for simple types (e.g., 4 for int)
                - A BaseModel instance
                - A dict that will be validated against the target type (partial updates allowed)

        Returns:
            The updated partial state.

        Examples:
            >>> builder = OutputBuilder(target=int)
            >>> builder.update(4)

            >>> builder = OutputBuilder(target=MyModel)
            >>> builder.update(MyModel(field="value"))
            >>> builder.update({"field": "value"})  # Partial update
        """
        # For BaseModel types, validate against the partial type (all-optional)
        # to allow incremental updates
        if isinstance(self.normalized, type) and issubclass(
            self.normalized, BaseModel
        ):
            adapter = TypeAdapter(self.partial_type)
        else:
            adapter = TypeAdapter(self.normalized)

        validated = adapter.validate_python(updates)

        # Handle different type scenarios
        if self._is_simple:
            # For simple types, just set the value directly
            self._partial = validated  # type: ignore
        elif isinstance(validated, BaseModel):
            # For BaseModel types, merge with existing partial
            if self._partial is None:
                self._partial = validated  # type: ignore
            elif isinstance(self._partial, BaseModel):
                # Merge by copying with updates
                self._partial = self._partial.model_copy(  # type: ignore
                    update=validated.model_dump(exclude_none=True)
                )
            else:
                self._partial = validated  # type: ignore

            # Track filled fields
            for name in self.field_names:
                if getattr(validated, name, None) is not None:
                    self._filled_fields.add(name)
        else:
            # For other types, set directly
            self._partial = validated

        # Add to history
        self._history.append(self._partial)  # type: ignore

        return self._partial  # type: ignore

    def update_from_pydantic_ai_result(
        self,
        result: PydanticAIAgentResult,
        fields: str | List[str] | None = None,
    ) -> Output:
        """Update the builder's state from a `pydantic_ai` AgentRunResult,
        this can update the entire output object or only a specified
        field or set of fields.

        Args:
            result: The AgentRunResult containing the output.
            fields: If set, update only these field(s). Can be a single field name
                   or a list of field names for selective updates.

        Returns:
            The updated partial state.
        """
        output = result.output

        # Normalize fields to a list
        field_names: List[str] | None = None
        if fields is not None:
            if isinstance(fields, str):
                field_names = [fields]
            else:
                field_names = fields

        # Field-level update
        if field_names:
            if self._partial is None:
                if self._is_simple:
                    raise ValueError(
                        "Cannot do field-level update on a simple type"
                    )
                if isinstance(self.normalized, type) and issubclass(
                    self.normalized, BaseModel
                ):
                    self._partial = self.partial_type()  # type: ignore
                else:
                    raise ValueError(
                        "Cannot do field-level update on non-BaseModel target"
                    )

            # Handle single or multiple field updates
            update_dict: dict[str, Any] = {}
            for field_name in field_names:
                # Extract value - split models wrap in a 'content' field
                if len(field_names) == 1:
                    field_value = getattr(output, "content", output)
                else:
                    # For multiple fields, expect the output to be a model with those fields
                    field_value = getattr(output, field_name, None)

                if field_value is not None:
                    update_dict[field_name] = field_value
                    self._filled_fields.add(field_name)

            if isinstance(self._partial, BaseModel):
                self._partial = self._partial.model_copy(update=update_dict)  # type: ignore

        # Full object update
        else:
            if isinstance(output, BaseModel) and isinstance(
                self._partial, BaseModel
            ):
                # Merge with existing partial
                self._partial = self._partial.model_copy(  # type: ignore
                    update=output.model_dump(exclude_none=True)
                )
            else:
                self._partial = output  # type: ignore

            # Track filled fields
            if isinstance(output, BaseModel):
                for name in self.field_names:
                    if getattr(output, name, None) is not None:
                        self._filled_fields.add(name)

        self._history.append(self._partial)  # type: ignore
        return self._partial  # type: ignore

    async def update_from_pydantic_ai_stream(
        self,
        stream: PydanticAIAgentStream,
        fields: str | List[str] | None = None,
        debounce_by: float | None = 0.1,
    ) -> AsyncIterator[Output]:
        """Update the builder's state from a `pydantic_ai` StreamedRunResult,
        this can update the entire output object or only a specified
        field or set of fields.

        Args:
            stream: The pydantic_ai StreamedRunResult.
            fields: If set, stream only these field(s). Can be a single field name
                   or a list of field names.
            debounce_by: Debounce interval in seconds.

        Yields:
            Progressively more complete Output instances.
        """
        # Normalize fields to a list
        field_names: List[str] | None = None
        if fields is not None:
            if isinstance(fields, str):
                field_names = [fields]
            else:
                field_names = fields

        async for output in stream.stream_output(debounce_by=debounce_by):
            # Field-level streaming
            if field_names:
                if self._partial is None:
                    if isinstance(self.normalized, type) and issubclass(
                        self.normalized, BaseModel
                    ):
                        self._partial = self.partial_type()  # type: ignore
                    else:
                        self._partial = output  # type: ignore

                # Handle single or multiple field updates
                if isinstance(self._partial, BaseModel):
                    update_dict: dict[str, Any] = {}
                    for field_name in field_names:
                        # Extract value - split models wrap in a 'content' field
                        if len(field_names) == 1:
                            field_value = getattr(output, "content", output)
                        else:
                            field_value = getattr(output, field_name, None)

                        if field_value is not None:
                            update_dict[field_name] = field_value

                    self._partial = self._partial.model_copy(  # type: ignore
                        update=update_dict
                    )

            # Full object streaming
            else:
                if isinstance(output, BaseModel) and isinstance(
                    self._partial, BaseModel
                ):
                    # Merge BaseModel updates
                    self._partial = self._partial.model_copy(  # type: ignore
                        update=output.model_dump(exclude_none=True)
                    )
                elif isinstance(self._partial, BaseModel):
                    # If partial is BaseModel but output is not, don't replace it
                    # This handles cases where output is a different type that will be
                    # handled separately (shouldn't happen in normal streaming)
                    pass
                else:
                    self._partial = output  # type: ignore

            yield self._partial if self._partial is not None else output

        # Record final state and track filled fields
        if self._partial is not None:
            if isinstance(self._partial, BaseModel):
                for name in self.field_names:
                    if getattr(self._partial, name, None) is not None:
                        self._filled_fields.add(name)
            self._history.append(self._partial)

    def finalize(self, *, exclude_none: bool = False) -> Output:
        """Return the final output, converting back to the original type if needed.

        For value-based targets (where target was an instance rather than a type),
        this method handles conversion:
        - dict targets: converts BaseModel back to dict
        - dataclass targets: converts BaseModel back to dataclass

        Args:
            exclude_none: If ``True``, omit fields whose value is ``None``
                from the serialized output.  Useful for selective edits where
                only the changed fields should appear in the result.

        Returns:
            The finalized output in its target type.

        Raises:
            ValueError: If no output has been produced yet.
        """
        if self._partial is None:
            raise ValueError("No output has been produced yet")

        # Convert back to original type if needed (dataclass, dict)
        if (
            self._is_value
            and isinstance(self.target, dict)
            and isinstance(self._partial, BaseModel)
        ):
            return self._partial.model_dump(exclude_none=exclude_none)

        if (
            self._is_value
            and dataclasses.is_dataclass(type(self.target))
            and isinstance(self._partial, BaseModel)
        ):
            dump = self._partial.model_dump(exclude_none=exclude_none)
            return type(self.target)(**dump)  # type: ignore

        return self._partial
