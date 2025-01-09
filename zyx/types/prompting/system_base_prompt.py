"""
### zyx.types.chat_completions.prompting.system_base_prompt

Module containing the base system prompt type used for chat completions.
"""

from __future__ import annotations

# [Imports]
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel

from .prompt import Prompt
from .prompt_section import PromptSection


class SystemBasePrompt(Prompt):
    """
    Base system prompt type used for chat completions.

    A system prompt is composed of multiple sections that can include:
    - Target: The target model or schema being used
    - Context: Additional context information
    - Guardrails: Any constraints or rules
    - Tools: Available function tools
    """

    def __init__(
        self,
        target: Union[Dict[str, Any], BaseModel, Type[BaseModel]],
        context: Optional[Any] = None,
        guardrails: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize a system prompt with target, optional context and tools.

        Args:
            target: The target model or schema, can be:
                - Dictionary of sections
                - Pydantic model class
                - Pydantic model instance
            context: Optional context information
            guardrails: Optional constraints or rules
            tools: Optional list of function tools
        """
        super().__init__()

        # Add target sections
        if isinstance(target, dict):
            for title, content in target.items():
                self.add_section(title, content)
        elif isinstance(target, (BaseModel, type)):
            self._add_model_section("Target", target)

        # Add context if provided
        if context is not None:
            self._add_context_section(context)

        # Add guardrails if provided
        if guardrails is not None:
            self._add_guardrails_section(guardrails)

        # Add tools if provided
        if tools is not None:
            self._add_tools_section(tools)

    def _add_model_section(self, title: str, model: Union[Type[BaseModel], BaseModel]) -> None:
        """Add a section for a Pydantic model class or instance."""
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

            self.add_section(title, "\n".join(lines))
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

            self.add_section(title, "\n".join(lines))

    def _add_context_section(self, context: Any) -> None:
        """Add a section for context information."""
        if isinstance(context, str):
            context_str = context
        else:
            try:
                import json

                context_str = json.dumps(context)
            except Exception:
                context_str = str(context)
        self.add_section("Context", context_str)

    def _add_guardrails_section(self, guardrails: Any) -> None:
        """Add a section for guardrails/constraints."""
        if isinstance(guardrails, str):
            guardrails_text = guardrails
        else:
            try:
                import json

                guardrails_text = json.dumps(guardrails)
            except Exception:
                guardrails_text = str(guardrails)
        self.add_section("Guardrails", guardrails_text)

    def _add_tools_section(self, tools: List[Dict[str, Any]]) -> None:
        """Add a section for function tools."""
        tool_lines = ["Available tools:"]

        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                name = func.get("name", "")
                desc = func.get("description", "No description provided")
                params = func.get("parameters", {})

                tool_lines.append(f"\n### {name}")
                tool_lines.append(desc)

                if "properties" in params:
                    tool_lines.append("\nParameters:")
                    for param_name, param_info in params["properties"].items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", f"The {param_name} parameter")
                        required = "required" if param_name in params.get("required", []) else "optional"
                        tool_lines.append(f"- **{param_name}** ({param_type}, {required}): {param_desc}")

        self.add_section("Tools", "\n".join(tool_lines))
