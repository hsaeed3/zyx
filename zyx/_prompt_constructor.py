"""
### zyx._prompt_constructor.py

Module containing the `PromptConstructor` class, which is used to construct prompts for tasks/etc.
"""

from __future__ import annotations

# [Imports]
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel

from .types.prompting.prompt import Prompt
from .types.prompting.system_base_prompt import SystemBasePrompt
from .types.prompting.prompt_section import PromptSection


class PromptConstructor:
    """
    A class for constructing various types of prompts with rich features.

    This class provides methods to build both basic prompts and system prompts
    with support for sections, models, context, guardrails, and tools.
    """

    def __init__(self) -> None:
        """Initialize an empty prompt constructor."""
        self._sections: List[PromptSection] = []
        self._metadata: Dict[str, Any] = {}

    def add_section(self, title: str, content: str) -> "PromptConstructor":
        """
        Add a new section to the prompt.

        Args:
            title: The title for the section
            content: The content text for the section

        Returns:
            PromptConstructor: Returns self for method chaining
        """
        self._sections.append(PromptSection(title=title, content=content))
        return self

    def add_model(self, model: Union[Type[BaseModel], BaseModel]) -> "PromptConstructor":
        """
        Add a Pydantic model as a formatted section.

        Args:
            model: The Pydantic model class or instance to add

        Returns:
            PromptConstructor: Returns self for method chaining
        """
        if isinstance(model, type):
            schema = model.model_json_schema()
            fields = schema.get("properties", {})
            required = schema.get("required", [])

            lines = [f"# {schema.get('title', model.__name__)} Schema"]
            if "description" in schema:
                lines.extend(["", schema["description"]])

            lines.append("\n## Fields")
            for field_name, field_info in fields.items():
                req = "(required)" if field_name in required else "(optional)"
                desc = field_info.get("description", "No description")
                type_str = field_info.get("type", "any")
                lines.append(f"- **{field_name}** ({type_str}) {req}: {desc}")
        else:
            model_name = model.__class__.__name__
            fields = model.model_fields

            lines = [f"# {model_name} Instance"]
            lines.append("\n## Fields and Values")

            for field_name, field in fields.items():
                field_type = (
                    field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
                )
                desc = field.description or f"The {field_name} value"
                value = getattr(model, field_name)
                lines.append(f"- **{field_name}** ({field_type}): {desc}")
                lines.append(f"  Current value: `{value}`")

        return self.add_section(model.__class__.__name__, "\n".join(lines))

    def add_dict(self, sections: Dict[str, str]) -> "PromptConstructor":
        """
        Add multiple sections from a dictionary.

        Args:
            sections: Dictionary mapping section titles to content

        Returns:
            PromptConstructor: Returns self for method chaining
        """
        for title, content in sections.items():
            self.add_section(title, content)
        return self

    def build_prompt(self) -> Prompt:
        """
        Build a basic Prompt object.

        Returns:
            Prompt: The constructed prompt with all added sections
        """
        return Prompt(sections=self._sections, metadata=self._metadata)

    def build_system_prompt(
        self,
        target: Union[Dict[str, Any], BaseModel, Type[BaseModel]],
        context: Optional[Any] = None,
        guardrails: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> SystemBasePrompt:
        """
        Build a SystemBasePrompt object with the specified components.

        Args:
            target: The target model or schema
            context: Optional context information
            guardrails: Optional constraints or rules
            tools: Optional list of function tools

        Returns:
            SystemBasePrompt: The constructed system prompt
        """
        return SystemBasePrompt(target=target, context=context, guardrails=guardrails, tools=tools)

    def set_metadata(self, key: str, value: Any) -> "PromptConstructor":
        """
        Set metadata for the prompt.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            PromptConstructor: Returns self for method chaining
        """
        self._metadata[key] = value
        return self
