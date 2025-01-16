"""
### zyx.core.types.tracing

This module contains types and interfaces for tracing and logging.
"""

from dataclasses import dataclass
from typing import Any, Optional
import random


# -----------------------------------------------------------------------------
# [TraceContext]
# -----------------------------------------------------------------------------


@dataclass
class TraceContext:
    """
    Represents the current context for traced events.
    """

    module: Optional[str] = None
    """The module name that is currently being traced."""

    entity: Optional[str] = None
    """The entity name that is currently being traced."""

    active: bool = False
    """Whether the context is currently active."""


# -----------------------------------------------------------------------------
# [Tracing Styles]
# (Color Codes for zyx)
# -----------------------------------------------------------------------------


class TracingStyles:
    """
    A class containing standard styles and colors used around the zyx library.
    """

    # [Color Codes]
    title_color: str = "bold light_sky_blue3"
    """The color code used for primary headers (tracing title, anywhere `zyx` is printed)"""

    subtitle_colors: list[str] = [
        "light_sea_green",
        "turqoise2",
        "spring_green1",
        "medium_spring_green",
        "cyan2",
        "pale_turquoise4",
        "steel_blue",
        "cornflower_blue",
        "aquamarine3",
    ]
    """Colors used for subtitles and secondary headers"""

    progress_colors: list[str] = ["green3", "spring_green3", "cyan3", "dark_turquoise", "turquoise2"]
    """Colors used for progress bars and loading indicators"""

    error_color: str = "red1"
    """Color used for error messages and warnings"""

    success_color: str = "green1"
    """Color used for success messages and checkmarks"""

    @classmethod
    def subtitle(cls, message: Any) -> str:
        """Get a random color suitable for subtitles"""
        color = random.choice(cls.subtitle_colors)
        return f"[{color}]{message}[/{color}]"

    @classmethod
    def title(cls, message: Any) -> str:
        """Format a message in the title style"""
        return f"[{cls.title_color}]{message}[/{cls.title_color}]"

    @classmethod
    def progress(cls, message: Any) -> str:
        """Format a progress message with a random progress color"""
        color = random.choice(cls.progress_colors)
        return f"[{color}]{message}[/{color}]"

    @classmethod
    def error(cls, message: Any) -> str:
        """Format an error message in red"""
        return f"[{cls.error_color}]{message}[/{cls.error_color}]"

    @classmethod
    def success(cls, message: Any) -> str:
        """Format a success message in green"""
        return f"[{cls.success_color}]{message}[/{cls.success_color}]"
    
    @classmethod
    def randomcolor(cls) -> str:
        colors = [
            "red", "green", "blue", "yellow", "magenta", "cyan", "white",
            "bright_red", "bright_green", "bright_blue", "bright_yellow", "bright_magenta", "bright_cyan",
            "light_sea_green", "turquoise2", "spring_green1", "medium_spring_green", "cyan2",
            "pale_turquoise4", "steel_blue", "cornflower_blue", "aquamarine3",
            "purple4", "deep_pink4", "hot_pink3", "pink3", "light_pink4",
            "dark_orange", "orange1", "light_goldenrod2", "wheat1", "navajo_white1",
            "khaki1", "light_goldenrod3", "light_yellow3", "gray62", "light_steel_blue",
            "dark_slate_gray2", "cadet_blue", "sky_blue1", "steel_blue1", "dark_turquoise",
            "green3", "spring_green3", "cyan3", "blue1", "purple1"
        ]
        return random.choice(colors)