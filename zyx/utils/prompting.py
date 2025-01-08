"""
zyx.core.logging.context

This module contains helper functions for converting various input object types
to a string context.
"""

from __future__ import annotations

# [Imports]
from typing import Any, Union, Literal
from pydantic import BaseModel
import json

from zyx import logging


# ==============================================================
# [Context String Creation]
# ==============================================================


def convert_object_to_prompt_context(
    context: Union[str, BaseModel, Any], 
    output_format: Literal["none", "markdown"] = "none"
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
                    field_type = field.annotation.__name__ if hasattr(field.annotation, '__name__') else str(field.annotation)
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
        raise logging.ZyxException(
            f"Failed to convert {context_type} {context} to JSON string context: {e}"
        )
