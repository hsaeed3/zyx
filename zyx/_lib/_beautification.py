"""zyx._lib._beautification"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from rich import box, get_console
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from ..core.memory.memory import Memory
    from ..core.memory.types import (
        MemoryItem,
        MemoryQueryResponse,
        MemorySearchResult,
    )
    from ..models.definition import ModelDefinition, ModelSettings
    from ..models.language.types import LanguageModelResponse


def _pretty_print_model(
    model: "ModelDefinition",
    kind: Literal["language_model", "embedding_model", "definition"],
) -> str:
    if kind == "language_model":
        content = "LanguageModel:\n"
    elif kind == "embedding_model":
        content = "EmbeddingModel:\n"
    elif kind == "definition":
        content = "ModelDefinition:\n"
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
    model: "ModelDefinition",
    kind: Literal["language_model", "embedding_model", "definition"],
):
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
    elif kind == "definition":
        model_name = "ModelDefinition"
        color_name = "bright_cyan"
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

    # Add settings using the settings' own rich print method
    if model.settings:
        from rich.console import RenderableType

        settings_renderable = model.settings.__rich__()
        # Add a spacer row then the settings
        output_table.add_row("", "")

    return Group(
        output_table,
        settings_renderable if model.settings else Text.from_markup(""),
        Text.from_markup(""),
    )


def _pretty_print_model_settings(settings: "ModelSettings") -> str:
    """Pretty print model settings as a simple string."""
    kind = settings.kind

    if kind == "language_model":
        content = "LanguageModelSettings:\n"
    elif kind == "embedding_model":
        content = "EmbeddingModelSettings:\n"
    else:
        content = "ModelSettings:\n"

    for key, value in settings.__dict__.items():
        content += f"\n>>> {key.replace('_', ' ').title()}: {value}"

    return content


def _rich_pretty_print_model_settings(settings: "ModelSettings"):
    """Pretty print model settings using rich with a rounded box table."""
    kind = settings.kind

    # Determine the color based on the kind
    if kind == "language_model":
        color_name = "dodger_blue2"
        settings_name = "LanguageModelSettings"
    elif kind == "embedding_model":
        color_name = "pale_green3"
        settings_name = "EmbeddingModelSettings"
    else:
        color_name = "bright_cyan"
        settings_name = "ModelSettings"

    # Create a table with rounded box
    settings_table = Table(
        show_edge=True,
        expand=False,
        row_styles=["none", "dim"],
        box=box.ROUNDED,
        border_style=color_name,
    )

    settings_table.add_column(
        f"[bold {color_name}]{settings_name}",
        style=f"dim bright_white",
        no_wrap=True,
    )
    settings_table.add_column("", justify="right", style="dim italic")

    # Add all settings as rows
    for key, value in settings.__dict__.items():
        settings_table.add_row(
            f"{key.replace('_', ' ').title()}:",
            f"{escape(str(value))}",
        )

    return settings_table


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


# ─────────────────────────────────────────────────────────────────────────────
# Memory pretty printing
# ─────────────────────────────────────────────────────────────────────────────


def _pretty_print_memory(memory: "Memory") -> str:
    content = "Memory:\n"
    content += f">>> Location: {memory.settings.location}\n"
    content += (
        f">>> Embeddings: {'Enabled' if memory.settings.embeddings else 'Disabled'}\n"
    )
    if memory.settings.embeddings and memory.settings.embedding_model:
        content += f">>> Embedding Model: {memory.settings.embedding_model.model}\n"
    content += f">>> Deduplicate: {memory.settings.deduplicate}\n"
    content += f">>> Table: {memory.settings.table_name}"
    return content


def _rich_pretty_print_memory(memory: "Memory"):
    output_table = Table(
        show_edge=False,
        expand=False,
        row_styles=["none", "dim"],
        box=box.SIMPLE,
    )
    output_table.add_column(
        f"\n[bold cyan]Memory:",
        style="bold cyan",
        no_wrap=True,
    )
    output_table.add_column("", justify="right")

    output_table.add_row(
        "\n\n[bold dim sandy_brown]Location:[/bold dim sandy_brown]",
        f"\n\n[bold dim italic]{escape(memory.settings.location)}[/bold dim italic]",
    )

    output_table.add_row(
        "[dim bright_white]Embeddings:[/dim bright_white]",
        f"[dim italic]{'Enabled' if memory.settings.embeddings else 'Disabled'}[/dim italic]",
    )

    if memory.settings.embeddings and memory.settings.embedding_model:
        output_table.add_row(
            "[dim bright_white]Embedding Model:[/dim bright_white]",
            f"[dim italic]{escape(memory.settings.embedding_model.model)}[/dim italic]",
        )

    output_table.add_row(
        "[dim bright_white]Deduplicate:[/dim bright_white]",
        f"[dim italic]{memory.settings.deduplicate}[/dim italic]",
    )

    output_table.add_row(
        "[dim bright_white]Table:[/dim bright_white]",
        f"[dim italic]{escape(memory.settings.table_name)}[/dim italic]",
    )

    return Group(output_table, Text.from_markup(""))


def _pretty_print_memory_item(item: "MemoryItem") -> str:
    content = "MemoryItem:\n"
    content += f">>> ID: {item.id}\n"
    content += (
        f">>> Content: {item.content[:100]}{'...' if len(item.content) > 100 else ''}\n"
    )
    if item.parsed is not None and item.parsed != item.content:
        content += f">>> Parsed: {item.parsed}\n"
    if item.metadata:
        content += f">>> Metadata: {item.metadata}\n"
    content += f">>> Created: {item.created_at.isoformat()}"
    return content


def _rich_pretty_print_memory_item(item: "MemoryItem"):
    output_table = Table(
        show_edge=False,
        expand=False,
        row_styles=["none", "dim"],
        box=box.SIMPLE,
    )
    output_table.add_column(
        f"\n[bold cyan]MemoryItem:",
        style="bold cyan",
        no_wrap=True,
    )
    output_table.add_column("", justify="right")

    output_table.add_row(
        "\n\n[bold dim sandy_brown]ID:[/bold dim sandy_brown]",
        f"\n\n[bold dim italic]{escape(item.id)}[/bold dim italic]",
    )

    content_display = item.content[:100] + ("..." if len(item.content) > 100 else "")
    output_table.add_row(
        "[dim bright_white]Content:[/dim bright_white]",
        f"[dim italic]{escape(content_display)}[/dim italic]",
    )

    if item.parsed is not None and str(item.parsed) != item.content:
        output_table.add_row(
            "[dim bright_white]Parsed:[/dim bright_white]",
            f"[dim italic]{escape(str(item.parsed))}[/dim italic]",
        )

    if item.metadata:
        output_table.add_row(
            "[dim bright_white]Metadata:[/dim bright_white]",
            f"[dim italic]{escape(str(item.metadata))}[/dim italic]",
        )

    output_table.add_row(
        "[dim bright_white]Created:[/dim bright_white]",
        f"[dim italic]{escape(item.created_at.isoformat())}[/dim italic]",
    )

    return Group(output_table, Text.from_markup(""))


def _pretty_print_memory_search_result(result: "MemorySearchResult") -> str:
    content = "MemorySearchResult:\n"
    content += f">>> Score: {result.score:.4f}\n"
    if result.distance is not None:
        content += f">>> Distance: {result.distance:.4f}\n"
    if result.model:
        content += f">>> Model: {result.model}\n"
    content += f"\n>>> Item:\n"
    content += f"    ID: {result.item.id}\n"
    content += f"    Content: {result.item.content[:80]}{'...' if len(result.item.content) > 80 else ''}"
    return content


def _rich_pretty_print_memory_search_result(result: "MemorySearchResult"):
    output_table = Table(
        show_edge=False,
        expand=False,
        row_styles=["none", "dim"],
        box=box.SIMPLE,
    )
    output_table.add_column(
        f"\n[bold cyan]MemorySearchResult:",
        style="bold cyan",
        no_wrap=True,
    )
    output_table.add_column("", justify="right")

    output_table.add_row(
        "\n\n[bold dim sandy_brown]Score:[/bold dim sandy_brown]",
        f"\n\n[bold dim italic]{result.score:.4f}[/bold dim italic]",
    )

    if result.distance is not None:
        output_table.add_row(
            "[dim bright_white]Distance:[/dim bright_white]",
            f"[dim italic]{result.distance:.4f}[/dim italic]",
        )

    if result.model:
        output_table.add_row(
            "[dim bright_white]Model:[/dim bright_white]",
            f"[dim italic]{escape(result.model)}[/dim italic]",
        )

    output_table.add_row(
        "[dim bright_white]Item ID:[/dim bright_white]",
        f"[dim italic]{escape(result.item.id)}[/dim italic]",
    )

    content_display = result.item.content[:80] + (
        "..." if len(result.item.content) > 80 else ""
    )
    output_table.add_row(
        "[dim bright_white]Content:[/dim bright_white]",
        f"[dim italic]{escape(content_display)}[/dim italic]",
    )

    return Group(output_table, Text.from_markup(""))


def _pretty_print_memory_query_response(response: "MemoryQueryResponse") -> str:
    content = "MemoryQueryResponse:\n"
    content += f">>> LLM Response: {response.response.content}\n"
    content += f">>> Model: {response.language_model}\n"
    content += f">>> Results Count: {len(response.results)}\n"
    if response.results:
        content += f"\n>>> Top Result:\n"
        content += f"    Score: {response.results[0].score:.4f}\n"
        content += f"    Content: {response.results[0].item.content[:60]}..."
    return content


def _rich_pretty_print_memory_query_response(response: "MemoryQueryResponse"):
    output_table = Table(
        show_edge=False,
        expand=False,
        row_styles=["none", "dim"],
        box=box.SIMPLE,
    )
    output_table.add_column(
        f"\n[bold cyan]MemoryQueryResponse:",
        style="bold cyan",
        no_wrap=True,
    )
    output_table.add_column("", justify="right")

    output_table.add_row(
        "",
        f"[italic]{escape(str(response.response.content))}[/italic]",
    )

    output_table.add_row(
        "\n[dim sandy_brown]Model:[/dim sandy_brown]",
        f"\n[dim italic]{escape(response.language_model)}[/dim italic]",
    )

    output_table.add_row(
        "[dim bright_white]Results Count:[/dim bright_white]",
        f"[dim italic]{len(response.results)}[/dim italic]",
    )

    if response.results:
        output_table.add_row(
            "[dim bright_white]Top Result Score:[/dim bright_white]",
            f"[dim italic]{response.results[0].score:.4f}[/dim italic]",
        )
        content_display = response.results[0].item.content[:60] + "..."
        output_table.add_row(
            "[dim bright_white]Top Result Content:[/dim bright_white]",
            f"[dim italic]{escape(content_display)}[/dim italic]",
        )

    return Group(output_table, Text.from_markup(""))
