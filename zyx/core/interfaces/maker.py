"""zyx.core.interfaces.maker"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Generic, List, Literal, Type, TypeVar

from instructor.dsl.simple_type import is_simple_type
from pydantic import BaseModel, Field, create_model

from ..._internal._exceptions import MakerError
from ..models.language.model import LanguageModel
from ..models.language.types import (
    LanguageModelName,
    LanguageModelResponse,
)
from ..processing.schemas.schema import Schema

_logger = logging.getLogger(__name__)


MakerT = TypeVar("MakerT")
"""The set output type of a `maker`."""


class Maker(Generic[MakerT]):
    """A `maker` is the backend interface that is used by the `zyx.make()` and
    `zyx.edit()` functions. You can initialize a maker on a per-type, or per-schema
    basis to get access to 'make()' and 'edit()' for a shared type.
    """

    @property
    def type(self) -> Type[MakerT]:
        """The type of the maker."""
        return self._schema.type

    @property
    def schema(self) -> Schema[MakerT]:
        """A 'Schema' representation of the type used to instantiate this maker."""
        return self._schema

    @property
    def model(self) -> LanguageModel[MakerT]:
        """The model of the maker."""
        return self._model

    def __init__(
        self,
        type: Type[MakerT] | Schema[MakerT] = str,
        model: LanguageModelName
        | LanguageModel[MakerT]
        | str = "openai/gpt-4o-mini",
        instructions: str | None = None,
        title: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
        key: str | None = None,
    ) -> None:
        """Initialize the maker with a type and model."""

        if not isinstance(type, Schema):
            self._schema = Schema(
                type,
                title=title,
                description=description,
                exclude=exclude,
                key=key,
            )
        else:
            self._schema = type

        if isinstance(model, str):
            model = LanguageModel(model, instructions=instructions)
        else:
            if instructions:
                model.instructions = instructions

        self._model = model
        self._instructions = instructions

    def _is_simple_type(self) -> bool:
        """Check if the type is a simple type (str, int, etc.) vs a complex model."""
        return not (
            isinstance(self._schema.source, type)
            and issubclass(self._schema.source, BaseModel)
        )

    def _create_batch_model(self, n: int) -> Type[BaseModel]:
        """Create a model that wraps the maker's type in a list for batch generation.

        Args:
            n: The number of items to generate

        Returns:
            A Pydantic model with a single field containing a list of the maker's type
        """
        from pydantic import conlist

        # Get the base type
        base_type = (
            self._schema.model
            if not self._is_simple_type()
            else self._schema.source
        )

        # Create a constrained list with exactly n items
        list_field = (
            conlist(base_type, min_length=n, max_length=n),  # type: ignore
            Field(
                ..., description=f"A list of exactly {n} generated items"
            ),
        )

        # Create the batch model
        batch_model = create_model(
            f"Batch{self._schema.title or 'Generation'}",
            items=list_field,
            __doc__=f"Batch of {n} {self._schema.title or 'items'}",
        )

        return batch_model

    def _build_generation_prompt(
        self,
        prompt: str | None,
        n: int,
        diversity: Literal["low", "medium", "high"] | None,
        has_context: bool = False,
    ) -> str:
        """Build a clear, hallucination-free prompt for generation.

        Args:
            prompt: User-provided generation context
            n: Number of items to generate
            diversity: Level of diversity in batch generation
            has_context: Whether context messages were provided

        Returns:
            A system prompt for the model
        """
        type_desc = (
            self._schema.description or f"type {self._schema.title}"
        )

        if n == 1:
            # Single generation
            if prompt:
                return f"Generate the requested {type_desc}."
            else:
                # No prompt - generate contextually appropriate data
                if has_context:
                    return f"Generate a contextually relevant {type_desc} based on the conversation."
                else:
                    return f"Generate a realistic, well-structured example of {type_desc}."
        else:
            # Batch generation
            diversity_guidance = {
                "low": "Generate variations that are similar in structure but distinct in details.",
                "medium": "Generate diverse examples covering different aspects or approaches.",
                "high": "Generate maximally diverse examples exploring different interpretations and possibilities.",
            }.get(diversity or "medium", "Generate diverse examples.")

            if prompt:
                return f"Generate {n} distinct examples of {type_desc}. {diversity_guidance}"
            else:
                # No prompt - generate diverse contextually appropriate data
                if has_context:
                    return f"Generate {n} contextually relevant examples of {type_desc} based on the conversation. {diversity_guidance}"
                else:
                    return f"Generate {n} realistic, well-structured examples of {type_desc}. {diversity_guidance}"

    def _build_confirmation_prompt(
        self,
        prompt: str | None,
        reason: bool,
    ) -> str:
        """Build a prompt for field confirmation step."""
        type_schema = json.dumps(self._schema.json, indent=2)

        base_prompt = f"Based on the request, determine which fields need to be generated.\n\nSchema:\n{type_schema}"

        if reason:
            base_prompt += "\n\nProvide reasoning for each field."

        return base_prompt

    def _create_field_confirmation_model(
        self, reason: bool = False
    ) -> Type[BaseModel]:
        """Create a model for confirming which fields need to be generated/edited."""
        if self._is_simple_type():
            # For simple types, create a single field confirmation
            fields = {
                "needs_generation": (
                    bool,
                    Field(..., description="Whether generation is needed"),
                )
            }
            if reason:
                fields["reason"] = (
                    str,
                    Field(..., description="Reasoning for this decision"),
                )
            return create_model("FieldConfirmation", **fields)
        else:
            # For complex models, create a field for each model field
            fields = {}
            for (
                field_name,
                field_info,
            ) in self._schema.model.model_fields.items():
                fields[field_name] = (
                    bool,
                    Field(
                        ...,
                        description=f"Whether {field_name} needs to be generated",
                    ),
                )
                if reason:
                    fields[f"{field_name}_reason"] = (
                        str,
                        Field(
                            ..., description=f"Reasoning for {field_name}"
                        ),
                    )
            return create_model("FieldConfirmation", **fields)

    def _create_edit_strategy_model(
        self, reason: bool = False
    ) -> Type[BaseModel]:
        """Create a model for determining edit strategy for each field."""
        if self._is_simple_type():
            # For simple types
            fields = {
                "needs_edit": (
                    bool,
                    Field(..., description="Whether editing is needed"),
                ),
                "strategy": (
                    Literal["replace", "modify"],
                    Field(
                        ...,
                        description="Edit strategy: 'replace' or 'modify'",
                    ),
                ),
            }
            if reason:
                fields["reason"] = (
                    str,
                    Field(..., description="Reasoning for this edit"),
                )
            return create_model("EditStrategy", **fields)
        else:
            # For complex models
            fields = {}
            for (
                field_name,
                field_info,
            ) in self._schema.model.model_fields.items():
                fields[f"{field_name}_needs_edit"] = (
                    bool,
                    Field(
                        ...,
                        description=f"Whether {field_name} needs editing",
                    ),
                )

                # Determine appropriate strategies based on field type
                field_type = field_info.annotation
                if hasattr(field_type, "__origin__"):
                    origin = field_type.__origin__
                    if origin is list:
                        strategy_type = Literal[
                            "replace", "append", "modify", "remove"
                        ]
                    elif origin is dict:
                        strategy_type = Literal[
                            "replace", "merge", "modify"
                        ]
                    else:
                        strategy_type = Literal["replace", "modify"]
                else:
                    strategy_type = Literal["replace", "modify"]

                fields[f"{field_name}_strategy"] = (
                    strategy_type,
                    Field(..., description=f"Strategy for {field_name}"),
                )

                if reason:
                    fields[f"{field_name}_reason"] = (
                        str,
                        Field(
                            ..., description=f"Reasoning for {field_name}"
                        ),
                    )

            return create_model("EditStrategy", **fields)

    def _process_messages(
        self,
        prompt: str | None,
        messages: List[Dict[str, Any]] | None,
    ) -> tuple[List[Dict[str, Any]], bool]:
        """Process prompt and messages into the format expected by the language model.

        Args:
            prompt: Optional string prompt
            messages: Optional list of message dicts for context

        Returns:
            Tuple of (processed message list, has_context flag)
        """
        processed = []

        # Add context messages (keep last 10 for context window efficiency)
        if messages:
            if len(messages) > 10:
                processed.extend(messages[-10:])
            else:
                processed.extend(messages)

        # Add prompt as final user message if provided
        if prompt:
            processed.append({"role": "user", "content": prompt})

        # Track whether we have context (messages or prompt)
        has_context = bool(messages) or bool(prompt)

        # If nothing provided, add a minimal trigger message
        if not processed:
            processed.append({"role": "user", "content": "Generate."})

        return processed, has_context

    async def amake(
        self,
        prompt: str | None = None,
        *,
        messages: List[Dict[str, Any]] | None = None,
        n: int = 1,
        confirm: bool = False,
        reason: bool = False,
        parallel: bool = False,
        diversity: Literal["low", "medium", "high"] | None = None,
    ) -> (
        LanguageModelResponse[MakerT] | LanguageModelResponse[List[MakerT]]
    ):
        """Asynchronously make/generate new instance(s) of the type.

        Args:
            prompt: Optional generation prompt/context (if None, generates contextually appropriate data)
            messages: Optional conversation history for context
            n: Number of instances to generate (default: 1)
            confirm: If True, first confirm which fields need to be generated
            reason: If True (requires confirm=True), include reasoning for decisions
            parallel: If True with n>1, generate each instance separately in parallel
            diversity: Level of diversity for batch generation ("low", "medium", "high")

        Returns:
            LanguageModelResponse with content as single item (n=1) or list (n>1)

        Raises:
            MakerError: If n < 1 or if generation fails
        """
        if n < 1:
            raise MakerError(f"n must be at least 1, got {n}")

        try:
            processed_messages, has_context = self._process_messages(
                prompt, messages
            )

            # Add instructions if available
            if self._instructions:
                instruction_msg = {
                    "role": "system",
                    "content": self._instructions,
                }
                if (
                    processed_messages
                    and processed_messages[0].get("role") == "system"
                ):
                    processed_messages[0]["content"] = (
                        f"{self._instructions}\n\n{processed_messages[0]['content']}"
                    )
                else:
                    processed_messages.insert(0, instruction_msg)

            # Handle confirmation step if requested
            if confirm:
                confirmation_model = self._create_field_confirmation_model(
                    reason=reason
                )
                confirmation_prompt = self._build_confirmation_prompt(
                    prompt, reason
                )

                confirmation_messages = processed_messages + [
                    {"role": "system", "content": confirmation_prompt}
                ]

                confirmation_response = await self._model.arun(
                    messages=confirmation_messages,
                    type=confirmation_model,
                )

                confirmation_data = confirmation_response.content

                # Check if generation is needed
                if self._is_simple_type():
                    if not confirmation_data.needs_generation:
                        return LanguageModelResponse(
                            content=None, raw=None
                        )
                else:
                    fields_to_generate = {
                        field_name
                        for field_name in self._schema.model.model_fields.keys()
                        if getattr(confirmation_data, field_name, False)
                    }

                    if not fields_to_generate:
                        return LanguageModelResponse(
                            content=None, raw=None
                        )

                    # Create schema with only fields that need generation
                    excluded_fields = (
                        set(self._schema.model.model_fields.keys())
                        - fields_to_generate
                    )
                    generation_schema = Schema(
                        self._schema.source,
                        title=self._schema.title,
                        description=self._schema.description,
                        exclude=excluded_fields,
                        key=self._schema.key,
                    )
            else:
                generation_schema = self._schema

            # Build generation prompt
            gen_prompt = self._build_generation_prompt(
                prompt, n, diversity, has_context
            )
            gen_messages = processed_messages + [
                {"role": "system", "content": gen_prompt}
            ]

            if is_simple_type(generation_schema.source):
                response_type = generation_schema.source
            else:
                response_type = generation_schema.model

            # Generate based on strategy
            if n == 1:
                # Single generation
                return await self._model.arun(
                    messages=gen_messages,
                    type=response_type,
                )
            elif not parallel:
                # Batch generation (single call with list type)
                batch_model = self._create_batch_model(n)

                response = await self._model.arun(
                    messages=gen_messages,
                    type=batch_model,
                )

                # Extract items from batch response
                items = response.content.items if response.content else []
                return LanguageModelResponse(
                    content=items, raw=response.raw
                )
            else:
                # Parallel generation (multiple separate calls)
                batch_messages = [gen_messages for _ in range(n)]

                responses = await self._model.abatch_run(
                    batch_messages=batch_messages,
                    type=response_type,
                )

                # Collect all generated items
                items = [
                    resp.content for resp in responses if resp.content
                ]

                # Return with the last response's raw data
                return LanguageModelResponse(
                    content=items,
                    raw=responses[-1].raw if responses else None,
                )
        except MakerError:
            # Re-raise MakerError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in MakerError
            raise MakerError(f"Generation failed: {e}") from e

    def make(
        self,
        prompt: str | None = None,
        *,
        messages: List[Dict[str, Any]] | None = None,
        n: int = 1,
        confirm: bool = False,
        reason: bool = False,
        parallel: bool = False,
        diversity: Literal["low", "medium", "high"] | None = None,
    ) -> (
        LanguageModelResponse[MakerT] | LanguageModelResponse[List[MakerT]]
    ):
        """Make/generate new instance(s) of the type (synchronous version).

        Args:
            prompt: Optional generation prompt/context (if None, generates contextually appropriate data)
            messages: Optional conversation history for context
            n: Number of instances to generate (default: 1)
            confirm: If True, first confirm which fields need to be generated
            reason: If True (requires confirm=True), include reasoning for decisions
            parallel: If True with n>1, generate each instance separately in parallel
            diversity: Level of diversity for batch generation ("low", "medium", "high")

        Returns:
            LanguageModelResponse with content as single item (n=1) or list (n>1)
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
                        confirm=confirm,
                        reason=reason,
                        parallel=parallel,
                        diversity=diversity,
                    ),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.amake(
                    prompt=prompt,
                    messages=messages,
                    n=n,
                    confirm=confirm,
                    reason=reason,
                    parallel=parallel,
                    diversity=diversity,
                )
            )

    async def aedit(
        self,
        value: MakerT,
        prompt: str | None = None,
        *,
        messages: List[Dict[str, Any]] | None = None,
        confirm: bool = False,
        selective: bool = False,
        reason: bool = False,
        parallel: bool = False,
    ) -> LanguageModelResponse[MakerT]:
        """Asynchronously edit an existing instance of the type.

        Args:
            value: The existing value to edit
            prompt: Optional edit instruction/context
            messages: Optional conversation history for context
            confirm: If True, first confirm which fields need to be edited
            selective: If True, determine edit strategy (merge/replace/append/etc) for each field
            reason: If True, include reasoning for edits
            parallel: If True, edit fields separately in parallel

        Returns:
            A LanguageModelResponse containing the edited content

        Raises:
            MakerError: If editing fails
        """
        try:
            processed_messages, _ = self._process_messages(
                prompt, messages
            )

            # Add current value context (cleanly formatted)
            if isinstance(value, BaseModel):
                value_str = value.model_dump_json(indent=2)
            else:
                value_str = str(value)

            # Insert value context before the last user message
            value_context_msg = {
                "role": "system",
                "content": f"Current value:\n{value_str}",
            }

            # Find insertion point (before last user message if exists)
            insert_idx = len(processed_messages)
            for i in range(len(processed_messages) - 1, -1, -1):
                if processed_messages[i].get("role") == "user":
                    insert_idx = i
                    break

            processed_messages.insert(insert_idx, value_context_msg)

            # Add instructions if available
            if self._instructions:
                instruction_msg = {
                    "role": "system",
                    "content": self._instructions,
                }
                if (
                    processed_messages
                    and processed_messages[0].get("role") == "system"
                ):
                    processed_messages[0]["content"] = (
                        f"{self._instructions}\n\n{processed_messages[0]['content']}"
                    )
                else:
                    processed_messages.insert(0, instruction_msg)

            # No confirmation - direct edit
            if not confirm and not selective:
                edit_prompt = "Update the value based on the request."
                edit_messages = processed_messages + [
                    {"role": "system", "content": edit_prompt}
                ]

                if is_simple_type(self._schema.source):
                    return await self._model.arun(
                        messages=edit_messages,
                        type=self._schema.source,
                    )
                return await self._model.arun(
                    messages=edit_messages,
                    type=self._schema.model,
                )

            # Confirmation/strategy step
            if selective:
                strategy_model = self._create_edit_strategy_model(
                    reason=reason
                )
                strategy_prompt = "Determine which fields need editing and the appropriate strategy for each."
                if reason:
                    strategy_prompt += " Provide reasoning."
            else:
                strategy_model = self._create_field_confirmation_model(
                    reason=reason
                )
                strategy_prompt = (
                    "Determine which fields need to be edited."
                )
                if reason:
                    strategy_prompt += " Provide reasoning."

            strategy_messages = processed_messages + [
                {"role": "system", "content": strategy_prompt}
            ]

            strategy_response = await self._model.arun(
                messages=strategy_messages,
                type=strategy_model,
            )

            strategy_data = strategy_response.content

            # Determine fields to edit
            if is_simple_type(self._schema.source):
                if selective:
                    if not strategy_data.needs_edit:
                        return LanguageModelResponse(
                            content=value, raw=None
                        )
                    edit_strategy = strategy_data.strategy
                else:
                    if not strategy_data.needs_generation:
                        return LanguageModelResponse(
                            content=value, raw=None
                        )
                    edit_strategy = "replace"
                fields_to_edit = set()
            else:
                fields_to_edit = set()
                field_strategies = {}

                for field_name in self._schema.model.model_fields.keys():
                    if selective:
                        needs_edit = getattr(
                            strategy_data,
                            f"{field_name}_needs_edit",
                            False,
                        )
                        if needs_edit:
                            fields_to_edit.add(field_name)
                            field_strategies[field_name] = getattr(
                                strategy_data,
                                f"{field_name}_strategy",
                                "replace",
                            )
                    else:
                        if getattr(strategy_data, field_name, False):
                            fields_to_edit.add(field_name)
                            field_strategies[field_name] = "replace"

                if not fields_to_edit:
                    return LanguageModelResponse(content=value, raw=None)

            # Perform the edit
            if not parallel or self._is_simple_type():
                # Edit all at once
                if fields_to_edit and selective:
                    strategy_info = "\n".join(
                        [
                            f"- {field}: {field_strategies[field]}"
                            for field in fields_to_edit
                        ]
                    )
                    edit_prompt = f"Update these fields:\n{strategy_info}\n\nProvide the complete updated object."
                else:
                    edit_prompt = "Update the value based on the request. Provide the complete updated object."

                edit_messages = processed_messages + [
                    {"role": "system", "content": edit_prompt}
                ]

                return await self._model.arun(
                    messages=edit_messages,
                    type=self._schema.model,
                )
            else:
                # Parallel field editing
                batch_messages = []
                field_names = list(fields_to_edit)

                for field_name in field_names:
                    field_info = self._schema.model.model_fields[
                        field_name
                    ]
                    current_field_value = (
                        getattr(value, field_name)
                        if isinstance(value, BaseModel)
                        else value
                    )
                    strategy = field_strategies.get(field_name, "replace")

                    field_prompt = f"Edit field '{field_name}' (current: {current_field_value}) using strategy: {strategy}"

                    batch_messages.append(
                        processed_messages
                        + [{"role": "system", "content": field_prompt}]
                    )

                # Create schemas for each field
                field_schemas = []
                for field_name in field_names:
                    field_info = self._schema.model.model_fields[
                        field_name
                    ]
                    field_model = create_model(
                        f"{field_name.capitalize()}Value",
                        **{
                            field_name: (field_info.annotation, field_info)
                        },
                    )
                    field_schemas.append(field_model)

                # Edit all fields in parallel
                responses = await self._model.abatch_run(
                    batch_messages=batch_messages,
                    type=field_schemas[0]
                    if len(field_schemas) == 1
                    else str,
                )

                # Collect field results
                field_results = {}
                for i, field_name in enumerate(field_names):
                    if len(field_schemas) > 1:
                        # Need individual calls since batch doesn't support varying types
                        field_response = await self._model.arun(
                            messages=batch_messages[i],
                            type=field_schemas[i],
                        )
                        field_results[field_name] = getattr(
                            field_response.content, field_name
                        )
                    else:
                        field_results[field_name] = getattr(
                            responses[i].content, field_name
                        )

                # Merge with original value
                if isinstance(value, BaseModel):
                    updated_data = value.model_dump()
                    updated_data.update(field_results)
                    final_content = self._schema.model(**updated_data)
                else:
                    final_content = field_results.get("value", value)

                return LanguageModelResponse(
                    content=final_content, raw=None
                )
        except MakerError:
            # Re-raise MakerError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in MakerError
            raise MakerError(f"Edit failed: {e}") from e

    def edit(
        self,
        value: MakerT,
        prompt: str | None = None,
        *,
        messages: List[Dict[str, Any]] | None = None,
        confirm: bool = False,
        selective: bool = False,
        reason: bool = False,
        parallel: bool = False,
    ) -> LanguageModelResponse[MakerT]:
        """Edit an existing instance of the type (synchronous version).

        Args:
            value: The existing value to edit
            prompt: Optional edit instruction/context
            messages: Optional conversation history for context
            confirm: If True, first confirm which fields need to be edited
            selective: If True, determine edit strategy (merge/replace/append/etc) for each field
            reason: If True, include reasoning for edits
            parallel: If True, edit fields separately in parallel

        Returns:
            A LanguageModelResponse containing the edited content
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.aedit(
                        value=value,
                        prompt=prompt,
                        messages=messages,
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
                    value=value,
                    prompt=prompt,
                    messages=messages,
                    confirm=confirm,
                    selective=selective,
                    reason=reason,
                    parallel=parallel,
                )
            )


async def amake(
    type: Type[MakerT] | Schema[MakerT] = str,
    prompt: str | None = None,
    *,
    n: int = 1,
    model: LanguageModelName
    | LanguageModel[MakerT]
    | str = "openai/gpt-4o-mini",
    instructions: str | None = None,
    messages: List[Dict[str, Any]] | None = None,
    title: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
    key: str | None = None,
    confirm: bool = False,
    reason: bool = False,
    parallel: bool = False,
    diversity: Literal["low", "medium", "high"] | None = None,
) -> LanguageModelResponse[MakerT] | LanguageModelResponse[List[MakerT]]:
    """Asynchronously make/generate new instance(s) of a given type.

    Parameters
    ----------
    type : Type[MakerT] | Schema[MakerT]
        The type of the maker
    prompt : str | None
        Optional generation prompt/context (if None, generates contextually appropriate data)
    n : int
        Number of instances to generate (default: 1)
    model : LanguageModelName | LanguageModel[MakerT] | str
        The model to use for generation
    instructions : str | None
        Optional instructions for the model
    title : str | None
        Optional title for the maker
    description : str | None
        Optional description for the maker
    exclude : set[str] | None
        Optional fields to exclude from the maker
    key : str | None
        Optional key for the maker
    confirm : bool
        If True, first confirm which fields need to be generated
    reason : bool
        If True (requires confirm=True), include reasoning for decisions
    parallel : bool
        If True with n>1, generate each instance separately in parallel
    diversity : Literal["low", "medium", "high"] | None
        Level of diversity for batch generation ("low", "medium", "high")

    Returns
    -------
    LanguageModelResponse[MakerT] | LanguageModelResponse[List[MakerT]]
        A LanguageModelResponse containing the generated content
    """
    return await Maker(
        type=type,
        model=model,
        instructions=instructions,
        title=title,
        description=description,
        exclude=exclude,
        key=key,
    ).amake(
        prompt=prompt,
        messages=messages,
        n=n,
        confirm=confirm,
        reason=reason,
        parallel=parallel,
        diversity=diversity,
    )


def make(
    type: Type[MakerT] | Schema[MakerT] = str,
    prompt: str | None = None,
    *,
    n: int = 1,
    model: LanguageModelName
    | LanguageModel[MakerT]
    | str = "openai/gpt-4o-mini",
    instructions: str | None = None,
    messages: List[Dict[str, Any]] | None = None,
    title: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
    key: str | None = None,
    confirm: bool = False,
    reason: bool = False,
    parallel: bool = False,
    diversity: Literal["low", "medium", "high"] | None = None,
) -> LanguageModelResponse[MakerT] | LanguageModelResponse[List[MakerT]]:
    """Make/generate new instance(s) of a given type.

    Parameters
    ----------
    type : Type[MakerT] | Schema[MakerT]
        The type of the maker
    prompt : str | None
        Optional generation prompt/context (if None, generates contextually appropriate data)
    n : int
        Number of instances to generate (default: 1)
    model : LanguageModelName | LanguageModel[MakerT] | str
        The model to use for generation
    instructions : str | None
        Optional instructions for the model
    title : str | None
        Optional title for the maker
    description : str | None
        Optional description for the maker
    exclude : set[str] | None
        Optional fields to exclude from the maker
    key : str | None
        Optional key for the maker
    confirm : bool
        If True, first confirm which fields need to be generated
    reason : bool
        If True (requires confirm=True), include reasoning for decisions
    parallel : bool
        If True with n>1, generate each instance separately in parallel
    diversity : Literal["low", "medium", "high"] | None
        Level of diversity for batch generation ("low", "medium", "high")

    Returns
    -------
    LanguageModelResponse[MakerT] | LanguageModelResponse[List[MakerT]]
        A LanguageModelResponse containing the generated content
    """
    return Maker(
        type=type,
        model=model,
        instructions=instructions,
        title=title,
        description=description,
        exclude=exclude,
        key=key,
    ).make(
        prompt=prompt,
        messages=messages,
        n=n,
        confirm=confirm,
        reason=reason,
        parallel=parallel,
        diversity=diversity,
    )


async def aedit(
    value: MakerT,
    prompt: str | None = None,
    *,
    model: LanguageModelName
    | LanguageModel[MakerT]
    | str = "openai/gpt-4o-mini",
    instructions: str | None = None,
    messages: List[Dict[str, Any]] | None = None,
    title: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
    key: str | None = None,
    confirm: bool = False,
    selective: bool = False,
    reason: bool = False,
    parallel: bool = False,
) -> LanguageModelResponse[MakerT]:
    """Asynchronously edit an existing instance, automatically inferring type from the value.

    Parameters
    ----------
    value : MakerT
        The existing value to edit
    prompt : str | None
        Optional edit instruction/context
    model : LanguageModelName | LanguageModel[MakerT] | str
        The model to use for editing
    instructions : str | None
        Optional instructions for the model
    messages : List[Dict[str, Any]] | None
        Optional conversation history for context
    title : str | None
        Optional title for the maker
    description : str | None
        Optional description for the maker
    exclude : set[str] | None
        Optional fields to exclude from editing
    key : str | None
        Optional key for the maker
    confirm : bool
        If True, first confirm which fields need to be edited
    selective : bool
        If True, determine edit strategy (merge/replace/append/etc) for each field
    reason : bool
        If True, include reasoning for edits
    parallel : bool
        If True, edit fields separately in parallel

    Returns
    -------
    LanguageModelResponse[MakerT]
        A LanguageModelResponse containing the edited content
    """
    # Infer type from the value
    inferred_type = type(value)

    return await Maker(
        type=inferred_type,
        model=model,
        instructions=instructions,
        title=title,
        description=description,
        exclude=exclude,
        key=key,
    ).aedit(
        value=value,
        prompt=prompt,
        messages=messages,
        confirm=confirm,
        selective=selective,
        reason=reason,
        parallel=parallel,
    )


def edit(
    value: MakerT,
    prompt: str | None = None,
    *,
    model: LanguageModelName
    | LanguageModel[MakerT]
    | str = "openai/gpt-4o-mini",
    instructions: str | None = None,
    messages: List[Dict[str, Any]] | None = None,
    title: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
    key: str | None = None,
    confirm: bool = False,
    selective: bool = False,
    reason: bool = False,
    parallel: bool = False,
) -> LanguageModelResponse[MakerT]:
    """Edit an existing instance, automatically inferring type from the value (synchronous version).

    Parameters
    ----------
    value : MakerT
        The existing value to edit
    prompt : str | None
        Optional edit instruction/context
    model : LanguageModelName | LanguageModel[MakerT] | str
        The model to use for editing
    instructions : str | None
        Optional instructions for the model
    messages : List[Dict[str, Any]] | None
        Optional conversation history for context
    title : str | None
        Optional title for the maker
    description : str | None
        Optional description for the maker
    exclude : set[str] | None
        Optional fields to exclude from editing
    key : str | None
        Optional key for the maker
    confirm : bool
        If True, first confirm which fields need to be edited
    selective : bool
        If True, determine edit strategy (merge/replace/append/etc) for each field
    reason : bool
        If True, include reasoning for edits
    parallel : bool
        If True, edit fields separately in parallel

    Returns
    -------
    LanguageModelResponse[MakerT]
        A LanguageModelResponse containing the edited content
    """
    # Infer type from the value
    inferred_type = type(value)

    return Maker(
        type=inferred_type,
        model=model,
        instructions=instructions,
        title=title,
        description=description,
        exclude=exclude,
        key=key,
    ).edit(
        value=value,
        prompt=prompt,
        messages=messages,
        confirm=confirm,
        selective=selective,
        reason=reason,
        parallel=parallel,
    )
