"""zyx._internal._beautification

makes stuff pretty. indeed my friend, beneath us is what we call negative performance.
why do we have it you ask? truly that is the question. but if i may, shall we not try to
yearn for the same beauty we see in the world around us?
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Literal

from rich import box
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

__all__ = ("Color", "ColorScheme", "Beautifier")


class Color(str, Enum):
    """Rich color palette for consistent styling across the library."""

    # Basic colors
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"

    # Bright variants
    BRIGHT_BLACK = "bright_black"
    BRIGHT_RED = "bright_red"
    BRIGHT_GREEN = "bright_green"
    BRIGHT_YELLOW = "bright_yellow"
    BRIGHT_BLUE = "bright_blue"
    BRIGHT_MAGENTA = "bright_magenta"
    BRIGHT_CYAN = "bright_cyan"
    BRIGHT_WHITE = "bright_white"

    # Blues
    DODGER_BLUE1 = "dodger_blue1"
    DODGER_BLUE2 = "dodger_blue2"
    DODGER_BLUE3 = "dodger_blue3"
    DEEP_SKY_BLUE1 = "deep_sky_blue1"
    STEEL_BLUE = "steel_blue"
    CORNFLOWER_BLUE = "cornflower_blue"
    ROYAL_BLUE1 = "royal_blue1"
    SKY_BLUE1 = "sky_blue1"

    # Greens
    SPRING_GREEN1 = "spring_green1"
    SPRING_GREEN2 = "spring_green2"
    SEA_GREEN1 = "sea_green1"
    PALE_GREEN1 = "pale_green1"
    PALE_GREEN3 = "pale_green3"
    LIGHT_GREEN = "light_green"
    CHARTREUSE1 = "chartreuse1"
    DARK_SEA_GREEN = "dark_sea_green"

    # Cyans & Turquoise
    DARK_TURQUOISE = "dark_turquoise"
    MEDIUM_TURQUOISE = "medium_turquoise"
    AQUAMARINE1 = "aquamarine1"
    DARK_CYAN = "dark_cyan"
    LIGHT_CYAN1 = "light_cyan1"

    # Purples & Violets
    PURPLE = "purple"
    MEDIUM_PURPLE = "medium_purple"
    MEDIUM_ORCHID = "medium_orchid"
    ORCHID = "orchid"
    PLUM2 = "plum2"
    VIOLET = "violet"
    BLUE_VIOLET = "blue_violet"

    # Oranges & Warm
    ORANGE1 = "orange1"
    DARK_ORANGE = "dark_orange"
    SANDY_BROWN = "sandy_brown"
    GOLD1 = "gold1"
    LIGHT_SALMON1 = "light_salmon1"
    LIGHT_CORAL = "light_coral"

    # Pinks & Reds
    HOT_PINK = "hot_pink"
    DEEP_PINK1 = "deep_pink1"
    PINK1 = "pink1"
    LIGHT_PINK1 = "light_pink1"
    INDIAN_RED1 = "indian_red1"

    # Grays
    GRAY37 = "gray37"
    GRAY50 = "gray50"
    GRAY63 = "gray63"
    GRAY69 = "gray69"
    GRAY84 = "gray84"


class ColorScheme:
    """A color scheme defining primary and accent colors for consistent theming.

    Attributes:
        primary: Main color used for titles and prominent elements
        accent: Secondary color used for highlights and important parameters
        border: Color for borders, outlines, and structural elements
        dim: Color for secondary/metadata information
    """

    def __init__(
        self,
        primary: Color | str,
        accent: Color | str,
        border: Color | str | None = None,
        dim: Color | str | None = None,
    ):
        # Convert Color enums to their string values
        if isinstance(primary, Color):
            self.primary = primary.value
        elif isinstance(primary, str):
            self.primary = primary
        else:
            self.primary = str(primary)

        if isinstance(accent, Color):
            self.accent = accent.value
        elif isinstance(accent, str):
            self.accent = accent
        else:
            self.accent = str(accent)

        if border is not None:
            if isinstance(border, Color):
                self.border = "gray63"
            elif isinstance(border, str):
                self.border = "gray63"
            else:
                self.border = "gray63"
        else:
            self.border = "gray63"

        if dim is not None:
            if isinstance(dim, Color):
                self.dim = dim.value
            elif isinstance(dim, str):
                self.dim = dim
            else:
                self.dim = str(dim)
        else:
            self.dim = Color.BRIGHT_WHITE.value


class Beautifier:
    """Unified interface for creating beautiful console output with consistent styling.

    The Beautifier provides factory methods for different types of objects (responses, models,
    settings, etc.) and handles both plain-text and rich rendering automatically.

    Example:
        ```python
        # Define your own color scheme
        scheme = ColorScheme(
            primary=Color.DODGER_BLUE2,
            accent=Color.SANDY_BROWN,
            border=Color.DODGER_BLUE3,
        )

        # Create a response printer
        beautifier = Beautifier.for_response(
            type_name="LanguageModelResponse",
            scheme=scheme,
            content=response.content,
            main_params={"Model": response.model},
            metadata={"Type": response.type, "Streamed": response.is_chunk}
        )

        # Get plain text output
        print(beautifier)

        # Get rich output
        print(beautifier.rich())
        ```
    """

    # Color system as class variable
    Colors: ClassVar[type[Color]] = Color
    Scheme: ClassVar[type[ColorScheme]] = ColorScheme

    def __init__(
        self,
        type_name: str,
        scheme: ColorScheme,
        layout: Literal["table", "boxed", "panel", "tree"] = "table",
    ):
        """Initialize a Beautifier instance.

        Args:
            type_name: Display name of the object type (e.g., "LanguageModelResponse")
            scheme: ColorScheme to use for styling
            layout: Visual layout style to use
        """
        self.type_name = type_name
        self.scheme = scheme
        self.layout = layout

        # Content storage
        self._content: Any = None
        self._content_style: str = "italic"
        self._main_params: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._nested: list[RenderableType] = []

    @classmethod
    def for_response(
        cls,
        type_name: str,
        scheme: ColorScheme,
        content: Any,
        main_params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Beautifier:
        """Factory method for creating response-style output.

        Response style emphasizes the main content with parameters below.

        Args:
            type_name: Name of the response type
            scheme: ColorScheme to use for styling
            content: Main content to display prominently
            main_params: Important parameters to highlight
            metadata: Additional metadata (will be dimmed)

        Returns:
            Configured Beautifier instance
        """
        beautifier = cls(type_name, scheme, layout="table")
        beautifier._content = content
        beautifier._main_params = main_params or {}
        beautifier._metadata = metadata or {}
        return beautifier

    @classmethod
    def for_model(
        cls,
        type_name: str,
        scheme: ColorScheme,
        main_params: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        nested: Any = None,
    ) -> Beautifier:
        """Factory method for creating model-style output.

        Model style focuses on parameters with optional nested objects (like settings).

        Args:
            type_name: Name of the model type
            scheme: ColorScheme to use for styling
            main_params: Important model parameters
            metadata: Additional metadata (will be dimmed)
            nested: Nested object(s) that will be rendered below (e.g., settings)

        Returns:
            Configured Beautifier instance
        """
        beautifier = cls(type_name, scheme, layout="table")
        beautifier._main_params = main_params or {}
        beautifier._metadata = metadata or {}
        if nested:
            if hasattr(nested, "__rich__"):
                beautifier._nested.append(nested.__rich__())
            elif hasattr(nested, "__str__"):
                beautifier._nested.append(Text(str(nested)))
        return beautifier

    @classmethod
    def for_settings(
        cls,
        type_name: str,
        scheme: ColorScheme,
        params: dict[str, Any],
        style: Literal["boxed", "inline"] = "boxed",
    ) -> Beautifier:
        """Factory method for creating settings-style output.

        Settings style shows key-value pairs in a bordered container.

        Args:
            type_name: Name of the settings type
            scheme: ColorScheme to use for styling
            params: All settings parameters to display
            style: Display style ("boxed" for bordered table, "inline" for simple list)

        Returns:
            Configured Beautifier instance
        """
        beautifier = cls(type_name, scheme, layout=style)
        beautifier._main_params = params
        return beautifier

    @classmethod
    def for_object(
        cls,
        type_name: str,
        scheme: ColorScheme,
        items: dict[str, Any],
        layout: Literal["table", "tree", "panel"] = "table",
    ) -> Beautifier:
        """Factory method for creating generic object output.

        Generic style for flexible object representation.

        Args:
            type_name: Name of the object type
            scheme: ColorScheme to use for styling
            items: All items to display
            layout: Visual layout style

        Returns:
            Configured Beautifier instance
        """
        beautifier = cls(type_name, scheme, layout=layout)
        beautifier._main_params = items
        return beautifier

    def __str__(self) -> str:
        """Generate plain-text representation."""
        lines = [f"{self.type_name}:"]

        # Add main content if present
        if self._content is not None:
            content_str = (
                str(self._content)
                if self._content
                else "No Output Content"
            )
            lines.append(content_str)

        # Add main parameters
        if self._main_params:
            if self._content is not None:
                lines.append("")  # Spacing after content
            for key, value in self._main_params.items():
                if value is not None:
                    lines.append(f">>> {key}: {value}")

        # Add metadata
        if self._metadata:
            for key, value in self._metadata.items():
                if value is not None:
                    lines.append(f">>> {key}: {value}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Generate plain-text representation."""
        return self.__str__()

    def rich(self) -> RenderableType:
        """Generate rich console representation."""
        if self.layout == "boxed":
            return self._render_boxed()
        elif self.layout == "tree":
            return self._render_tree()
        elif self.layout == "panel":
            return self._render_panel()
        else:  # table (default)
            return self._render_table()

    def __rich__(self) -> RenderableType:
        """Rich console protocol support."""
        return self.rich()

    def _render_table(self) -> RenderableType:
        """Render as a table layout (default for responses and models)."""
        table = Table(
            show_edge=False,
            show_header=True,
            expand=False,
            row_styles=["none", "dim"],
            box=box.SIMPLE,
        )

        # Add header column with type name
        table.add_column(
            f"\n[bold {self.scheme.primary}]{self.type_name}:",
            style=f"bold {self.scheme.primary}",
            no_wrap=True,
        )
        table.add_column("", justify="right")

        # Add main content if present
        if self._content is not None:
            content_text = escape(str(self._content))

            # Render content as syntax inside a panel spanning both columns
            content_syntax = Syntax(
                content_text,
                lexer="markdown",
                word_wrap=True,
                dedent=True,
            )
            content_panel = Panel(
                content_syntax,
                border_style=self.scheme.border,
                padding=(0, 0),
                expand=True,
                highlight=True,
            )
            table.add_row(
                "",  # empty first cell
                content_panel,
                end_section=True,  # ensures full width
            )

        # Add main parameters
        for key, value in self._main_params.items():
            if value is not None:
                label = f"\n[bold dim {self.scheme.accent}]{key}:[/bold dim {self.scheme.accent}]"
                val = f"\n[bold dim italic]{escape(str(value))}[/bold dim italic]"
                table.add_row(label, val)

        # Add metadata parameters
        for key, value in self._metadata.items():
            if value is not None:
                label = f"[dim {self.scheme.dim}]{key}:[/dim {self.scheme.dim}]"
                val = f"[dim italic]{escape(str(value))}[/dim italic]"
                table.add_row(label, val)

        # If there are nested objects, group them
        if self._nested:
            return Group(table, *self._nested, Text.from_markup(""))

        return Group(table, Text.from_markup(""))

    def _render_boxed(self) -> RenderableType:
        """Render as a boxed table (typically for settings)."""
        table = Table(
            show_edge=True,
            expand=False,
            row_styles=["none", "dim"],
            box=box.ROUNDED,
            border_style=self.scheme.border,
        )

        # Title column
        table.add_column(
            f"[bold {self.scheme.primary}]{self.type_name}",
            style=f"dim {self.scheme.dim}",
            no_wrap=True,
        )
        table.add_column("", justify="right", style="dim italic")

        # Add all parameters
        all_params = {**self._main_params, **self._metadata}
        for key, value in all_params.items():
            if value is not None:
                # Format key nicely
                formatted_key = key.replace("_", " ").title()
                table.add_row(f"{formatted_key}:", escape(str(value)))

        return table

    def _render_tree(self) -> RenderableType:
        """Render as a tree structure."""
        tree = Tree(
            f"[bold {self.scheme.primary}]{self.type_name}[/bold {self.scheme.primary}]"
        )

        # Add main content if present
        if self._content is not None:
            content_node = tree.add(
                f"[{self._content_style}]{escape(str(self._content))}[/{self._content_style}]"
            )

        # Add main parameters
        if self._main_params:
            params_node = tree.add(
                f"[bold {self.scheme.accent}]Parameters[/bold {self.scheme.accent}]"
            )
            for key, value in self._main_params.items():
                if value is not None:
                    params_node.add(
                        f"[dim]{key}:[/dim] {escape(str(value))}"
                    )

        # Add metadata
        if self._metadata:
            meta_node = tree.add(
                f"[dim {self.scheme.dim}]Metadata[/dim {self.scheme.dim}]"
            )
            for key, value in self._metadata.items():
                if value is not None:
                    meta_node.add(
                        f"[dim]{key}:[/dim] [dim italic]{escape(str(value))}[/dim italic]"
                    )

        return tree

    def _render_panel(self) -> RenderableType:
        """Render as a panel."""
        lines = []

        # Main content
        if self._content is not None:
            lines.append(
                f"[{self._content_style}]{escape(str(self._content))}[/{self._content_style}]"
            )
            lines.append("")

        # Main parameters
        for key, value in self._main_params.items():
            if value is not None:
                lines.append(
                    f"[bold {self.scheme.accent}]{key}:[/bold {self.scheme.accent}] "
                    f"[bold dim]{escape(str(value))}[/bold dim]"
                )

        # Metadata
        for key, value in self._metadata.items():
            if value is not None:
                lines.append(
                    f"[dim]{key}:[/dim] [dim italic]{escape(str(value))}[/dim italic]"
                )

        content = Text.from_markup("\n".join(lines))
        return Panel(
            content,
            title=f"[bold {self.scheme.primary}]{self.type_name}[/bold {self.scheme.primary}]",
            border_style=self.scheme.border,
            expand=False,
        )
