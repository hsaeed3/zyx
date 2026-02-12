"""zyx.operations.edit.strategy"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Literal, Type, Generic, TypeVar

from pydantic import BaseModel, Field, create_model

from ..._processing._outputs import (
    partial_output_model,
    sparse_output_model,
    selection_output_model,
    split_output_model_by_fields,
    split_output_model,
)
from ..._utils._outputs import OutputBuilder


Output = TypeVar("Output")


@dataclass
class AbstractEditStrategy(ABC, Generic[Output]):
    """Abstract base class for type-specific strategies used by the
    `edit` semantic operation."""

    class Confirmation(BaseModel):
        needs_edits: bool = Field(
            ...,
            description="Determination if any edits need to be made to the source data provided to you.",
        )

    builder: OutputBuilder[Output]

    @property
    @abstractmethod
    def kind(self) -> str:
        pass

    @abstractmethod
    def get_replacement_schema(self) -> Type[BaseModel]:
        raise NotImplementedError

    @abstractmethod
    def get_selective_schema(self) -> Type[BaseModel]:
        raise NotImplementedError

    @abstractmethod
    def get_plan_schema(self) -> Type[BaseModel]:
        raise NotImplementedError

    @abstractmethod
    def get_plan_edits_schema(self, plan: BaseModel | Any) -> List[Type[BaseModel]]:
        raise NotImplementedError

    @abstractmethod
    def get_plan_edit_schema(self, plan: BaseModel | Any) -> Type[BaseModel]:
        raise NotImplementedError

    def apply_replacement(self, replacement: Any) -> Output:
        raise NotImplementedError

    def apply_selective(self, edits: Any) -> Output:
        raise NotImplementedError

    def apply_plan_edits(self, plan: BaseModel, edits: Any) -> Output:
        raise NotImplementedError


@dataclass
class BasicEditStrategy(AbstractEditStrategy[Output]):
    """Fallback strategy for non-text, non-mapping outputs.

    Contract:
    - Replacement: provide a complete new value of the normalized type.
    - Selective: same as Replacement (no sub-fields to select).
    - Plan: choose between 'null' (no edits) and 'replace' (apply one replacement).
    """

    @property
    def kind(self) -> Literal["basic"]:
        return "basic"

    def get_replacement_schema(self) -> Type[BaseModel]:
        """
        For basic types, the replacement is just a full new value of the normalized type.
        We wrap it in a single-field Pydantic model so the instructions/docs are clear.
        """
        normalized = (
            self.builder.partial_type
        )  # already handles simple/BaseModel
        model = create_model(
            "Replacement",
            value=(
                normalized,
                Field(
                    ...,
                    description=(
                        "A complete replacement value for the data provided to you. "
                        "Return the fully updated value here."
                    ),
                ),
            ),
        )
        return model

    def get_selective_schema(self) -> Type[BaseModel]:
        """
        For scalar/basic outputs, 'selective' edits have the same shape as replacement:
        you either provide a new value or nothing.
        """
        # You can just reuse the same schema:
        return self.get_replacement_schema()

    def get_plan_schema(self) -> Type[BaseModel]:
        """
        Simple plan: either perform no edits ('null') or replace the value ('replace').
        """
        Plan = create_model(
            "EditPlan",
            selection=(
                Literal["null", "replace"],
                Field(
                    ...,
                    description=(
                        "Plan for editing this value. "
                        "'null' = no changes are needed. "
                        "'replace' = you will provide a complete replacement value."
                    ),
                ),
            ),
        )
        return Plan

    def get_plan_edits_schema(self, plan: BaseModel) -> List[Type[BaseModel]]:
        """
        If the plan says 'replace', we expect one replacement value.
        If 'null', no edits are required.
        """
        if getattr(plan, "selection") == "replace":
            return [self.get_replacement_schema()]
        return []

    def get_plan_edit_schema(self, plan: BaseModel) -> Type[BaseModel]:
        """
        Single-edit case: if plan == 'replace', use the replacement schema;
        if 'null', you won't actually be asked for edits.
        """
        return self.get_replacement_schema()

    def apply_replacement(self, replacement: Any) -> Output:
        value = getattr(replacement, "value", replacement)
        return self.builder.update(value)

    def apply_selective(self, edits: Any) -> Output:
        return self.apply_replacement(edits)

    def apply_plan_edits(self, plan: BaseModel, edits: Any) -> Output:
        if getattr(plan, "selection", None) == "null":
            if self.builder.partial is None and self.builder.is_value:
                return self.builder.update(self.builder.target)
            return self.builder.partial  # type: ignore[return-value]
        return self.apply_replacement(edits)


@dataclass
class TextEditStrategy(AbstractEditStrategy[str]):
    """Strategy for edits made to strings."""

    def __post_init__(self) -> None:
        if not self.builder.normalized == str:
            raise ValueError(
                "TextEditStrategy can only be used with string targets."
            )

    class Replacement(BaseModel):
        """A complete replacement for a piece of text/a string. All content
        within the text or string will be replaced with this replacement."""

        text: str = Field(
            ..., description="The text/content to replace the section with."
        )

    class Edit(BaseModel):
        """A single edit operation to be applied onto a string.

        This represents a single edit, and must include a start and end
        anchor which represents the starting and ending snippets of the
        section that is being edited."""

        start_anchor: str = Field(
            ...,
            description="The starting text/content of the section that is being edited.",
        )

        end_anchor: str = Field(
            ...,
            description="The ending text/content of the section that is being edited.",
        )

        replacement: str = Field(
            ..., description="The text/content to replace the section with."
        )

    class Edits(BaseModel):
        """A plan for a series of edit operations to be applied onto text
        content."""

        edits: List[TextEditStrategy.Edit] = Field(
            ...,
            description="The list of edit operations to be applied onto the text content.",
        )

    class Selection(BaseModel):
        """A selection of anchors to be edited within a string/text content. This
        is used to mark the starting and ending snippets of a specific section of
        content that is being edited."""

        start_anchor: str = Field(
            ...,
            description="The starting text/content of the section that is being edited.",
        )

        end_anchor: str = Field(
            ...,
            description="The ending text/content of the section that is being edited.",
        )

    class EditPlan(BaseModel):
        """A plan for a series of edit operations to be applied onto text content. Once
        you have selected the appropriate anchors for the context that will be replaced,
        you will be able to provide a replacement for the selected section(s).

        If not content needs to be edited, or replaced only return the 'null' option."""

        selections: List[TextEditStrategy.Selection] | Literal["null"] = Field(
            ...,
            description="The list of selections of anchors to be edited within the text content.",
        )

    @property
    def kind(self) -> Literal["text"]:
        return "text"

    def get_replacement_schema(self) -> Type[BaseModel]:
        return self.Replacement

    def get_selective_schema(self) -> Type[BaseModel]:
        return self.Edits

    def get_plan_schema(self) -> Type[BaseModel]:
        return self.EditPlan

    def get_plan_edit_schema(
        self, plan: BaseModel | Any
    ) -> Type[BaseModel]:
        """
        Dynamically builds a Pydantic model with one field per selection in the plan,
        using keys 'edit_1', 'edit_2', ... Each field includes a reference to the start/end anchor
        in its docstring.
        """
        fields = {}
        for idx, selection in enumerate(plan.selections, 1):
            field_name = f"edit_{idx}"
            field_doc = (
                f"Replacement for the selection spanning:\n"
                f"Start anchor (index {idx}): '{selection.start_anchor}'\n"
                f"End anchor (index {idx}): '{selection.end_anchor}'"
            )
            fields[field_name] = (
                self.get_replacement_schema(),
                Field(..., description=field_doc),
            )

        model_name = "Edits"
        docstring = (
            "Enumerated fields ('edit_1', 'edit_2', ...) corresponding to each selection "
            "within the plan. Each field contains a replacement for the section of text "
            "identified by its start/end anchors, as described in the field."
        )
        new_model = create_model(model_name, **fields)
        new_model.__doc__ = docstring
        return new_model

    def get_plan_edits_schema(
        self, plan: BaseModel | Any
    ) -> List[Type[BaseModel]]:
        schemas = []
        for edit in plan.selections:
            schemas.append(self.get_replacement_schema())

        return schemas

    def _base_text(self) -> str:
        if self.builder.partial is not None:
            return self.builder.partial
        if self.builder.is_value and isinstance(self.builder.target, str):
            return self.builder.target
        raise ValueError("Text edits require a string target value.")

    def _apply_anchor_edit(
        self,
        text: str,
        *,
        start_anchor: str,
        end_anchor: str,
        replacement: str,
    ) -> str:
        start_idx = text.find(start_anchor)
        if start_idx == -1:
            return text
        end_idx = text.find(end_anchor, start_idx + len(start_anchor))
        if end_idx == -1:
            return text
        end_idx += len(end_anchor)
        return text[:start_idx] + replacement + text[end_idx:]

    def apply_replacement(self, replacement: Any) -> str:
        text = getattr(replacement, "text", replacement)
        return self.builder.update(text)

    def apply_selective(self, edits: Any) -> str:
        base = self._base_text()
        if isinstance(edits, BaseModel) and hasattr(edits, "edits"):
            edit_list = edits.edits
        else:
            edit_list = edits
        updated = base
        for edit in edit_list:
            updated = self._apply_anchor_edit(
                updated,
                start_anchor=edit.start_anchor,
                end_anchor=edit.end_anchor,
                replacement=edit.replacement,
            )
        return self.builder.update(updated)

    def apply_plan_edits(self, plan: BaseModel, edits: Any) -> str:
        selections = getattr(plan, "selections", None)
        if selections in (None, "null"):
            if self.builder.partial is None and self.builder.is_value:
                return self.builder.update(self.builder.target)
            return self.builder.partial  # type: ignore[return-value]

        base = self._base_text()
        updated = base
        # edits may be a single model with edit_1... or a list of replacements
        if isinstance(edits, BaseModel):
            edits_map = {
                name: getattr(edits, name)
                for name in getattr(edits, "model_fields", {}).keys()
            }
            for idx, selection in enumerate(selections, 1):
                repl = edits_map.get(f"edit_{idx}")
                if repl is None:
                    continue
                replacement_text = getattr(repl, "text", repl)
                updated = self._apply_anchor_edit(
                    updated,
                    start_anchor=selection.start_anchor,
                    end_anchor=selection.end_anchor,
                    replacement=replacement_text,
                )
        else:
            for selection, repl in zip(selections, edits):
                replacement_text = getattr(repl, "text", repl)
                updated = self._apply_anchor_edit(
                    updated,
                    start_anchor=selection.start_anchor,
                    end_anchor=selection.end_anchor,
                    replacement=replacement_text,
                )

        return self.builder.update(updated)

    def apply_plan_edit_at_index(
        self, plan: "TextEditStrategy.EditPlan", index: int, edit: Any
    ) -> str:
        selections = getattr(plan, "selections", None)
        if selections in (None, "null"):
            if self.builder.partial is None and self.builder.is_value:
                return self.builder.update(self.builder.target)
            return self.builder.partial  # type: ignore[return-value]

        if not isinstance(selections, list) or index >= len(selections):
            return self.builder.partial  # type: ignore[return-value]

        selection = selections[index]
        replacement_text = getattr(edit, "text", edit)
        base = self._base_text()
        updated = self._apply_anchor_edit(
            base,
            start_anchor=selection.start_anchor,
            end_anchor=selection.end_anchor,
            replacement=replacement_text,
        )
        return self.builder.update(updated)


class MappingEditStrategy(AbstractEditStrategy[Output]):
    """Strategy for edits made to dictionaries/pydantic models/other mapping-like
    structures and types."""

    @property
    def kind(self) -> Literal["mapping"]:
        return "mapping"

    def get_replacement_schema(self) -> Type[BaseModel]:
        model = partial_output_model(self.builder.normalized)  # type: ignore[arg-type]
        doc = model.__doc__

        if doc is None:
            doc = ""
        doc += (
            "A complete replacement for mapping-like data that has been provided to you."
            "Only fields that should be replaced should be present."
        )
        model.__doc__ = doc
        model.__name__ = "Replacement"

        return model

    def get_selective_schema(self) -> Type[BaseModel]:
        model = sparse_output_model(self.builder.normalized)
        doc = model.__doc__

        if doc is None:
            doc = ""
        doc += (
            "A selective list of changes to be applied to the mapping-like data that has been provided to you."
            "Only fields that should be changed should be present."
        )
        model.__doc__ = doc
        model.__name__ = "Edits"

        return model

    def get_plan_schema(self) -> Type[BaseModel]:
        model = selection_output_model(self.builder.normalized)

        selections = self.builder.field_names + ["null", "all"]

        model = selection_output_model(
            selections, multi_select=True, name="EditPlan"
        )
        model.__doc__ = (
            "A plan for a series of edit operations to be applied onto the mapping-like data that has been provided to you."
            "Provide only the names of the fields that should be edited or replaced. If a field can be left as None, or as is, do not"
            "include it in the plan."
            "If no fields should be edited, selected, or replaced, select the 'null' option, and vice-versa with the `all` option."
        )
        return model

    def get_plan_edits_schema(self, plan: BaseModel | Any) -> List[Type[BaseModel]]:
        schemas = []

        selections = self._extract_plan_selections(plan)
        if not selections:
            return schemas

        split_models = split_output_model_by_fields(self.builder.normalized)
        for field, model in split_models.items():
            if field in selections:
                schemas.append(model)

        return schemas

    def get_plan_edit_schema(self, plan: BaseModel) -> Type[BaseModel]:
        selections = self._extract_plan_selections(plan)
        schema = split_output_model(
            self.builder.normalized,  # type: ignore[arg-type]
            selections,
            partial=False,
        )

        doc = schema.__doc__
        if doc is None:
            doc = ""

        doc += (
            "Based on the edit plan you determined was appropriate, you have been provided a sparse representation"
            "of the fields representing the changes that should be made to the source data. Provide an edit for"
            "all fields provided."
        )
        schema.__doc__ = doc
        schema.__name__ = "Edits"

        return schema

    def _extract_plan_selections(self, plan: BaseModel) -> List[str]:
        selections = None
        if hasattr(plan, "selections"):
            selections = plan.selections
        elif hasattr(plan, "indices"):
            indices = plan.indices
            choices = getattr(plan.__class__, "_choices", None) or getattr(
                plan, "_choices", None
            )
            if choices is not None:
                selections = [choices[i] for i in indices]  # type: ignore[arg-type]
            else:
                selections = indices

        if selections in (None, "null"):
            return []

        # normalize special options
        if isinstance(selections, list):
            if "all" in selections:
                return self.builder.field_names
            if "null" in selections:
                return []
            return [s for s in selections if isinstance(s, str)]

        return []

    def apply_replacement(self, replacement: Any) -> Output:
        if isinstance(replacement, BaseModel):
            updates = replacement.model_dump(exclude_none=True)
        elif isinstance(replacement, dict):
            updates = {k: v for k, v in replacement.items() if v is not None}
        else:
            updates = replacement
        return self.builder.update(updates)

    def apply_selective(self, edits: Any) -> Output:
        if isinstance(edits, BaseModel) and hasattr(edits, "changes"):
            updates: dict[str, Any] = {}
            for change in edits.changes:
                field = getattr(change, "field", None)
                value = getattr(change, "value", None)
                if field is not None:
                    updates[field] = value
            return self.builder.update(updates)

        if isinstance(edits, dict):
            return self.builder.update(edits)
        return self.builder.update(edits)

    def apply_plan_edits(self, plan: BaseModel, edits: Any) -> Output:
        selections = self._extract_plan_selections(plan)
        if not selections:
            if self.builder.partial is None and self.builder.is_value:
                return self.builder.update(self.builder.target)
            return self.builder.partial  # type: ignore[return-value]

        if isinstance(edits, BaseModel):
            updates = edits.model_dump(exclude_none=True)
            return self.builder.update(updates)
        if isinstance(edits, dict):
            return self.builder.update(edits)
        return self.builder.update(edits)


@dataclass(init=False)
class EditStrategy(AbstractEditStrategy[Output]):
    """
    Helper factory that auto-determines the appropriate edit strategy class given an
    `OutputBuilder` object.
    """

    @classmethod
    def create(
        cls, builder: OutputBuilder[Output]
    ) -> AbstractEditStrategy[Output]:

        if builder.field_count > 0:
            return MappingEditStrategy(builder)

        if builder.normalized == str:
            return TextEditStrategy(builder)  # type: ignore[arg-type]

        return BasicEditStrategy(builder)
