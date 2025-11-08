"""zyx.utils.beautification

... various utils and resources and stuff that make
zyx look good and beautiful by providing __rich__
and pretty print methods for various interfaces
and object types within `zyx`. Most 'main' interfaces within the
library have a specialized __rich__ method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..models.embeddings.model import EmbeddingModel

    from ..models.language.model import LanguageModel
    from ..models.language.types import LanguageModelResponse


def _pretty_print_model(
    model: EmbeddingModel | LanguageModel,
    kind: Literal["language_model", "embedding_model"],
) -> str:
    if kind == "language_model":
        content = "LanguageModel:\n"
    elif kind == "embedding_model":
        content = "EmbeddingModel:\n"
    else:
        raise ValueError(f"Invalid kind: {kind}")

    content.join(
        [
            f"\n>>> Model: {model.model}",
            f"\n>>> Backend Client Kind: {model.get_client().name}",
        ]
    )

    return content


def _rich_pretty_print_model(
    model: EmbeddingModel | LanguageModel,
    kind: Literal["language_model", "embedding_model"],
):
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.console import Group
    from rich.markup import escape

    output_table = Table(
        show_edge=False,
        expand=False,
        row_styles=["none", "dim"],
        box=box.SIMPLE,
    )

    if kind == "language_model":
        model_name = "LanguageModel"
        color_name = "dodger_blue2"
    elif kind == "embedding_model":
        model_name = "EmbeddingModel"
        color_name = "pale_green3"
    else:
        raise ValueError(f"Invalid kind: {kind}")

    output_table.add_column(
        f"\n[bold {color_name}]{model_name}:",
        style=f"bold {color_name}",
        no_wrap=True,
    )
    output_table.add_column("", justify="right")

    output_table.add_row(
        "\n\n[bold dim sandy_brown]Model:[/bold dim sandy_brown]",
        f"\n\n[bold dim italic]{escape(model.model)}[/bold dim italic]",
    )

    output_table.add_row(
        "[dim bright_white]Backend Client Kind:[/dim bright_white]",
        f"[dim italic]{escape(model.get_client().name)}[/dim italic]",
    )

    if kind == "embedding_model":
        output_table.add_row(
            "[dim bright_white]Dimensions:[/dim bright_white]",
            f"[dim italic]{escape(str(model.dimensions))}[/dim italic]",
        )
        output_table.add_row(
            "[dim bright_white]Encoding Format:[/dim bright_white]",
            f"[dim italic]{escape(str(model.encoding_format))}[/dim italic]",
        )
        output_table.add_row(
            "[dim bright_white]User:[/dim bright_white]",
            f"[dim italic]{escape(str(model.user))}[/dim italic]",
        )
    elif kind == "language_model":
        for key, value in model.settings.__dict__.items():
            output_table.add_row(
                f"[dim bright_white]{key.replace('_', ' ').title()}:[/dim bright_white]",
                f"[dim italic]{escape(str(value))}[/dim italic]",
            )

    return Group(output_table, Text.from_markup(""))


def _pretty_print_language_model_response(response: LanguageModelResponse) -> str:
    content = "LanguageModelResponse:\n"
    content += (
        f"{response.content if response.content is not None else 'No Output Content'}"
    )

    content += f"\n\n>>> Model: {response.model}"

    if response.content and not isinstance(response.content, str):
        content += f"\n>>> Type: {response.type}"
        content += f"\n>>> Instructor Mode: {response.instructor_mode.name}"

    if response.is_chunk:
        content += "\n>>> Streamed Response: True"

    # NOTE: we check for 'is_structured' to avoid printing the default response models or
    # provided response models used if Mode.TOOLS or a similar instructor mode that uses
    # function calling to generate a structured output.
    if response.has_tool_calls and not response.is_structured:
        content += f"\n>>> Tools Called: {', '.join(response.tools_called)}"
    return content


def _rich_pretty_print_language_model_response(response: LanguageModelResponse):
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.console import Group
    from rich.markup import escape

    output_table = Table(
        show_edge=False,
        show_header=True,
        expand=False,
        row_styles=["none", "dim"],
        box=box.SIMPLE,
    )
    output_table.add_column(
        f"\n[bold dodger_blue2]LanguageModelResponse:",
        style="bold dodger_blue2",
        no_wrap=True,
    )
    output_table.add_column("", justify="right")

    output_table.add_row("", f"[italic]{escape(str(response.content))}[/italic]")

    output_table.add_row(
        "\n[dim sandy_brown]Model:[/dim sandy_brown]",
        f"\n[dim italic]{escape(response.model)}[/dim italic]",
    )

    if response.content and not isinstance(response.content, str):
        output_table.add_row(
            "[dim bright_white]Type:[/dim bright_white]",
            f"[dim italic]{escape(str(response.type))}[/dim italic]",
        )
        output_table.add_row(
            "[dim bright_white]Instructor Mode:[/dim bright_white]",
            f"[dim italic]{escape(response.instructor_mode.name)}[/dim italic]",
        )

    if response.is_chunk:
        output_table.add_row(
            "[dim bright_white]Streamed Response:[/dim bright_white]",
            "[dim italic]True[/dim italic]",
        )
    if response.has_tool_calls and not response.is_structured:
        output_table.add_row(
            "[dim bright_white]Tools Called:[/dim bright_white]",
            f"[dim italic]{escape(', '.join(response.tools_called))}[/dim italic]",
        )
    return Group(output_table, Text.from_markup(""))
