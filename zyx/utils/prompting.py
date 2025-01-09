"""
zyx.core.logging.context

This module contains helper functions for converting various input object types
to a string context.
"""

from __future__ import annotations

# [Imports]
from typing import Any, Union, Literal, List, Dict, Optional, Type
from pydantic import BaseModel
import json

from ..types.prompting.prompt import Prompt
from ..types.prompting.system_base_prompt import SystemBasePrompt
from .._prompt_constructor import PromptConstructor
from zyx import logging


# ==============================================================
# [Context String Creation]
# ==============================================================


def convert_object_to_prompt_context(
    context: Union[str, BaseModel, Any], output_format: Literal["none", "markdown"] = "none"
) -> str:
    """
    Converts various input object types to a string context, with optional markdown output_formatting.

    Args:
        context: Object to convert to string context. Can be:
            - String (returned as-is)
            - Pydantic BaseModel class (converted to JSON schema)
            - Pydantic BaseModel instance (converted to JSON)
            - Other objects (converted to JSON string)
        output_format: Output output_format, either "none" for raw JSON or "markdown" for output_formatted markdown

    Returns:
        str: The context as a string, optionally output_formatted as markdown

    Raises:
        ZyxException: If the object cannot be converted to a string context
    """

    def log_and_return(context_string: str) -> str:
        logging.logger.debug("context string from %s is %s", context, context_string)
        return context_string

    try:
        # Handle raw string
        if isinstance(context, str):
            if output_format == "markdown":
                return f"```text\n{context}\n```"
            return context

        # Handle Pydantic model class
        if isinstance(context, type) and issubclass(context, BaseModel):
            schema = context.model_json_schema()
            if output_format == "markdown":
                # Extract field info from schema
                fields = schema.get("properties", {})
                required = schema.get("required", [])

                lines = [f"# {schema.get('title', context.__name__)} Schema", ""]
                if "description" in schema:
                    lines.extend([schema["description"], ""])

                lines.append("## Fields")
                for field_name, field_info in fields.items():
                    req = "(required)" if field_name in required else "(optional)"
                    desc = field_info.get("description", "No description")
                    type_str = field_info.get("type", "any")
                    lines.append(f"- **{field_name}** ({type_str}) {req}: {desc}")

                lines.extend(["", "```json", json.dumps(schema, indent=2), "```"])
                return "\n".join(lines)

            logging.logger.debug("schema: %s", schema)

            return log_and_return(json.dumps(schema))

        # Handle Pydantic model instance
        if isinstance(context, BaseModel):
            json_data = context.model_dump_json()
            if output_format == "markdown":
                model_name = context.__class__.__name__
                fields = context.model_fields

                lines = [f"# {model_name} Object", ""]
                lines.append(f"A {model_name} model with the following fields:")

                for field_name, field in fields.items():
                    field_type = (
                        field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
                    )
                    desc = field.description or f"The {field_name} value"
                    lines.append(f"- {field_name} ({field_type}): {desc}")

                lines.extend(["", "```json", json_data, "```"])
                return "\n".join(lines)

            logging.logger.debug("json_data: %s", json_data)

            return log_and_return(json_data)

        # Handle other objects
        json_str = json.dumps(context)
        if output_format == "markdown":
            type_name = type(context).__name__

            logging.logger.debug("type_name: %s", type_name)
            logging.logger.debug("json_str: %s", json_str)

            return f"# {type_name} Object\n\n```json\n{json_str}\n```"

        logging.logger.debug("json_str: %s", json_str)

        return log_and_return(json_str)

    except Exception as e:
        context_type = "model class" if isinstance(context, type) else "object"
        raise logging.ZyxException(f"Failed to convert {context_type} {context} to JSON string context: {e}")


def construct_model_prompt(model: Union[Type[BaseModel], BaseModel]) -> str:
    """
    Constructs a formatted prompt string from a Pydantic model class or instance.

    Args:
        model: Either a Pydantic BaseModel class or an instance of a BaseModel

    Returns:
        str: A formatted string containing the model's schema information and/or instance data

    Raises:
        ZyxException: If the model cannot be converted to a prompt string
    """
    try:
        # Handle model class
        if isinstance(model, type) and issubclass(model, BaseModel):
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

            return "\n".join(lines)

        # Handle model instance
        if isinstance(model, BaseModel):
            model_name = model.__class__.__name__
            fields = model.model_fields

            lines = [f"# {model_name} Instance"]
            lines.append("\n## Fields and Values")

            # Add field descriptions
            for field_name, field in fields.items():
                field_type = (
                    field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
                )
                desc = field.description or f"The {field_name} value"
                value = getattr(model, field_name)
                lines.append(f"- **{field_name}** ({field_type}): {desc}")
                lines.append(f"  Current value: `{value}`")

            return "\n".join(lines)

        raise logging.ZyxException(f"Input must be a Pydantic model class or instance, got {type(model)}")

    except Exception as e:
        model_type = "model class" if isinstance(model, type) else "model instance"
        raise logging.ZyxException(f"Failed to construct model prompt from {model_type}: {e}")


def construct_system_prompt(
    target: Union[Dict[str, Any], BaseModel, Type[BaseModel]],
    context: Optional[Any] = None,
    guardrails: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Prompt:
    """
    Constructs a system prompt from a target object with optional context and tools.

    Args:
        target: The target model or schema
        context: Optional context information
        guardrails: Optional constraints or rules
        tools: Optional list of function tools

    Returns:
        Prompt: The constructed prompt with all sections
    """
    try:
        constructor = PromptConstructor()

        # Add target sections
        if isinstance(target, dict):
            constructor.add_dict(target)
        elif isinstance(target, (BaseModel, type)) and (isinstance(target, type) or isinstance(target, BaseModel)):
            constructor.add_section("Target", construct_model_prompt(target))

        # Add context if provided
        if context is not None:
            context_str = convert_object_to_prompt_context(context)
            constructor.add_section("Context", context_str)

        # Add guardrails if provided
        if guardrails is not None:
            if isinstance(guardrails, str):
                guardrails_text = guardrails
            else:
                guardrails_text = convert_object_to_prompt_context(guardrails)
            constructor.add_section("Guardrails", guardrails_text)

        # Add tools if provided
        if tools is not None:
            tool_lines = ["Available tools:"]

            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    name = func.get("name", "")
                    desc = func.get("description", "No description provided")
                    params = func.get("parameters", {})

                    # Add function name and description
                    tool_lines.append(f"\n### {name}")
                    tool_lines.append(desc)

                    # Add parameter details if present
                    if "properties" in params:
                        tool_lines.append("\nParameters:")
                        for param_name, param_info in params["properties"].items():
                            param_type = param_info.get("type", "any")
                            param_desc = param_info.get("description", f"The {param_name} parameter")
                            required = "required" if param_name in params.get("required", []) else "optional"
                            tool_lines.append(f"- **{param_name}** ({param_type}, {required}): {param_desc}")

            constructor.add_section("Tools", "\n".join(tool_lines))

        return constructor.build_prompt()

    except Exception as e:
        target_type = (
            "model class"
            if isinstance(target, type)
            else "model instance"
            if isinstance(target, BaseModel)
            else "dictionary"
        )
        raise logging.ZyxException(f"Failed to construct system prompt with context and tools from {target_type}: {e}")
