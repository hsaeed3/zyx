"""zyx.utils.prettify"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import RenderableType, Group
from rich.text import Text
from rich.rule import Rule

if TYPE_CHECKING:
    from ..results import Result

__all__ = ("prettify_result",)


def prettify_result(result: Result) -> RenderableType:
    """
    'prettify' a `Result` object into a rich `Renderable` object,
    which pretty prints the result to the console when printed by
    `rich.print()`.
    """
    renderables: list[RenderableType] = []

    renderables.append(Rule(title="✨ Result", style="rule.line", align="left"))

    output_text = f"{result.output}\n"
    renderables.append(Text(output_text))

    renderables.append(
        Text.from_markup(
            f"[sandy_brown]>>>[/sandy_brown] [dim italic]Operation: {result.kind.capitalize()}[/dim italic]"
        )
    )

    if result.models:
        model_names = (
            result.models[0] if len(result.models) == 1 else ", ".join(result.models)
        )
        label = "Model" if len(result.models) == 1 else "Models"
        model_info = Text.from_markup(
            f"[sandy_brown]>>>[/sandy_brown] [dim italic]{label}: {model_names}[/dim italic]"
        )
        renderables.append(model_info)

    return Group(*renderables)
