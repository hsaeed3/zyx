"""
### zyx.types.chat_completions.prompting.prompt_section
"""

from __future__ import annotations

# [Imports]
from typing import Dict, Any, Mapping, Optional
from ..subscriptable_base_model import SubscriptableBaseModel


class PromptSection(SubscriptableBaseModel):
    """
    A section of a prompt, containing a title, content & 'editable' items, which can
    be directly regenerated for just the section.
    """

    title: Optional[str] = None
    content: str
    editable: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PromptSection:
        """
        Create a new `PromptSection` from a dictionary.
        """
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the `PromptSection` to a dictionary.
        """
        return self.model_dump()

    def __str__(self, title: str = "##", divider: str = "\n") -> str:
        """
        Convert the `PromptSection` to a string representation.

        Args:
            title: The title prefix to use. Defaults to "##".
            divider: The divider to use between sections. Defaults to newline.

        Returns:
            str: The string representation of the section with title, content and editable items.
        """
        # Build base string with title and content
        result = f"{self.title}{divider}{self.content}"

        # Add editable items if present
        if self.editable:
            result += f"{divider}Editable Items:{divider}"
            for key, value in self.editable.items():
                result += f"{key}: {value}{divider}"

        return result
