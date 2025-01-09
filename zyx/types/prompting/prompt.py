"""
### zyx.types.chat_completions.prompt

Module containing the base prompt type, as well as types for user & system prompts.
"""

from __future__ import annotations

# [Imports]
from typing import List
from .prompt_section import PromptSection
from ..subscriptable_base_model import SubscriptableBaseModel


class Prompt(SubscriptableBaseModel):
    """
    Base type for all instruction or task focused prompts in `zyx`.

    A prompt is composed of multiple sections, each containing a title, content and optional
    editable items. The sections are rendered in order when converting to string.
    """

    sections: List[PromptSection] = []

    def __str__(self, section_title_prefix: str = "##", section_divider: str = "\n\n") -> str:
        """
        Convert the prompt to a string representation.

        Args:
            section_title_prefix: The prefix to use for section titles. Defaults to "##".
            section_divider: The divider to use between sections. Defaults to double newline.

        Returns:
            str: The full prompt as a string with all sections rendered.
        """
        return section_divider.join(
            section.__str__(title=section_title_prefix, divider=section_divider) for section in self.sections
        )

    def add_section(self, title: str, content: str, editable: dict = None) -> None:
        """
        Add a new section to the prompt.

        Args:
            title: The title for the section
            content: The content text for the section
            editable: Optional dictionary of editable items for the section
        """
        self.sections.append(PromptSection(title=title, content=content, editable=editable))

    def get_section(self, title: str) -> PromptSection:
        """
        Get a section by its title.

        Args:
            title: The title of the section to retrieve

        Returns:
            PromptSection: The matching section if found

        Raises:
            KeyError: If no section with the given title exists
        """
        for section in self.sections:
            if section.title == title:
                return section
        raise KeyError(f"No section found with title: {title}")
