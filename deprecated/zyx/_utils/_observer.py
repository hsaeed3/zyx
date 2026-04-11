"""zyx._utils._observer

Lightweight Rich-based observation utilities for CLI progress and tool events.
"""

from __future__ import annotations

from typing import Any
from collections.abc import AsyncIterable


class Observer:
    """
    Rich-backed observer for CLI progress and tool events.

    Provides clean, visually appealing output using Rich panels and formatting.
    """

    def __init__(self, *, console: Any | None = None) -> None:
        from rich.console import Console

        self._console = console or Console()
        self._current_operation: str | None = None

    def emit(self, message: str) -> None:
        self._console.print(message)

    def _emit_panel(
        self,
        title: str,
        content: str,
        *,
        style: str = "bold",
        emoji: str = "",
    ) -> None:
        from rich.panel import Panel

        full_title = f"{emoji} {title}" if emoji else title
        panel = Panel(
            content, title=full_title, border_style=style, expand=False
        )
        self._console.print(panel)

    def on_node(self, node: Any, _ctx: Any | None = None) -> None:
        """Handle node-level events from agent.iter()."""
        try:
            from pydantic_ai import _agent_graph as _agent_graph
            from pydantic_ai import messages as _messages
            from pydantic_graph import End
        except Exception:
            return

        if isinstance(node, End):
            return

        if isinstance(node, _agent_graph.UserPromptNode):
            return

        if isinstance(node, _agent_graph.ModelRequestNode):
            return

        if isinstance(node, _agent_graph.CallToolsNode):
            tool_names: list[str] = []
            for part in node.model_response.parts or []:
                if isinstance(part, _messages.ToolCallPart):
                    tool_names.append(part.tool_name)
            if tool_names:
                self.on_tools_called(tool_names)
            return

    async def event_stream_handler(
        self, _run_ctx: Any, events: AsyncIterable[Any]
    ) -> None:
        """Handle streaming tool events from run_stream()."""
        from pydantic_ai import messages as _messages

        async for event in events:
            if isinstance(event, _messages.FunctionToolCallEvent):
                self.on_tool_call(event.part.tool_name)
            elif isinstance(event, _messages.FunctionToolResultEvent):
                self.on_tool_result(event.result.tool_name)

    def on_operation_start(self, operation: str) -> None:
        """Called when a semantic operation begins."""
        self._current_operation = operation
        self._emit_panel(
            "Operation",
            f"[bold white]{operation.title()}[/bold white]",
            style="bold cyan",
            emoji="▶",
        )

    def on_operation_complete(self, operation: str) -> None:
        """Called when a semantic operation completes."""
        self.emit(f"[dim]✓ {operation.title()} complete[/dim]")
        if self._current_operation == operation:
            self._current_operation = None

    def on_tools_called(self, tool_names: list[str]) -> None:
        """Called when multiple tools are invoked at once."""
        if not tool_names:
            return
        tools_str = "\n".join(
            f"  • [yellow]{name}[/yellow]" for name in tool_names
        )
        self._emit_panel(
            "Tools Called",
            tools_str,
            style="bold yellow",
            emoji="🔧",
        )

    def on_tool_call(self, tool_name: str) -> None:
        """Called when a single tool is invoked during streaming."""
        self.emit(f"  [yellow]🔧[/yellow] [bold]{tool_name}[/bold]")

    def on_tool_result(self, tool_name: str) -> None:
        """Called when a tool result is received during streaming."""
        self.emit(f"  [dim]↳[/dim] [italic]{tool_name}[/italic]")

    def on_fields_selected(self, fields: list[str]) -> None:
        """Called when fields are selected for editing."""
        if not fields:
            return
        fields_str = "\n".join(f"  • [blue]{field}[/blue]" for field in fields)
        self._emit_panel(
            "Fields Selected",
            fields_str,
            style="bold blue",
            emoji="✏️",
        )

    def on_fields_generated(
        self, fields: list[dict[str, Any]] | None = None
    ) -> None:
        """Called when fields are generated during editing."""
        if not fields:
            return
        fields_str = "\n".join(
            f"  • [green]{field['name']}[/green]" for field in fields
        )
        self._emit_panel(
            "Fields Generated",
            fields_str,
            style="bold green",
            emoji="✨",
        )

    def on_task_complete(self, summary: str) -> None:
        """Called when a run task completes."""
        self._emit_panel(
            "Task Complete",
            f"[white]{summary}[/white]",
            style="bold green",
            emoji="✓",
        )

    def on_verification(self, passed: bool, error: str | None = None) -> None:
        """Called when verification completes."""
        if passed:
            self.emit("  [green]✓[/green] [bold]Verification passed[/bold]")
        else:
            self.emit("  [red]✗[/red] [bold]Verification failed[/bold]")
            if error:
                self.emit(f"    [dim red]{error}[/dim red]")

    def on_model_request(self) -> None:
        """Called when making a request to the model."""
        suffix = (
            f" ({self._current_operation})" if self._current_operation else ""
        )
        self.emit(
            f"  [dim cyan]⟳[/dim cyan] [italic]Processing{suffix}...[/italic]"
        )
