"""zyx.stream"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    List,
    TypeVar,
)

from pydantic import BaseModel

from ._aliases import (
    PydanticAIAgentResult,
    PydanticAIAgentStream,
    PydanticAIUsage,
)
from ._utils._outputs import OutputBuilder
from .result import Result


Output = TypeVar("Output")


@dataclass
class Stream(Generic[Output]):
    """
    Streamed result wrapper for semantic operations.

    A `Stream` wraps the `output` of a semantic operation, which represents either a
    complete or partial representation of the `target` type or value, based on
    the operation used.

    This class provides both sync and async interfaces for streaming outputs.
    Sync methods block and collect, async methods stream in real-time.
    """

    _builder: OutputBuilder[Output] = field(repr=False)
    _streams: List[PydanticAIAgentStream] = field(repr=False)
    _field_mappings: List[Dict[str, Any]] = field(repr=False)
    _stream_contexts: List[Any] = field(repr=False)
    _exclude_none: bool = field(default=False, repr=False)
    """When ``True``, ``finalize()`` will omit ``None``-valued fields from
    the result (useful for selective edits where only changed fields should
    appear)."""

    _current_stream_index: int = field(default=0, init=False, repr=False)
    _raw: List[PydanticAIAgentResult[Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _is_complete: bool = field(default=False, init=False, repr=False)

    async def __aenter__(self) -> Stream[Output]:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._is_complete:
            await self.finish_async()

        for stream_ctx in self._stream_contexts:
            try:
                await stream_ctx.__aexit__(None, None, None)
            except Exception:
                pass

    async def stream_text(
        self, *, delta: bool = False, debounce_by: float | None = 0.1
    ) -> AsyncIterator[str]:
        """
        Stream the text output.

        Args:
            delta: Whether to stream the delta of the text output.
            debounce_by: The debounce interval in seconds.

        Yields:
            The text output of the stream.
        """
        for idx, stream in enumerate(self._streams):
            self._current_stream_index = idx
            mapping = self._field_mappings[idx]

            async for text_chunk in stream.stream_text(
                delta=delta, debounce_by=debounce_by
            ):
                yield text_chunk

            final_output = await stream.get_output()
            result = PydanticAIAgentResult(output=final_output)

            if mapping["update_output"]:
                self._builder.update_from_pydantic_ai_result(
                    result, fields=mapping["fields"]
                )

            self._raw.append(result)

        self._is_complete = True

    async def stream_partial(
        self, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[Output]:
        """
        Stream real-time updates of the target output. This method aggregates all runs
        taking place within the semantic operation, and is able to provide updates for
        multiple fields within the output.

        Args:
            debounce_by: The debounce interval in seconds.

        Yields:
            The partial output of the stream.
        """
        for idx, stream in enumerate(self._streams):
            self._current_stream_index = idx
            mapping = self._field_mappings[idx]

            if mapping["update_output"]:
                async for (
                    partial
                ) in self._builder.update_from_pydantic_ai_stream(
                    stream=stream,
                    fields=mapping["fields"],
                    debounce_by=debounce_by,
                ):
                    yield partial
            else:
                async for _ in stream.stream_output(debounce_by=debounce_by):
                    pass

            final_output = await stream.get_output()
            self._raw.append(PydanticAIAgentResult(output=final_output))

        self._is_complete = True

    async def stream_field(
        self, field_name: str, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[Any]:
        """
        Stream a specific field from the output.

        Args:
            field_name: The name of the field to stream.
            debounce_by: The debounce interval in seconds.

        Yields:
            The value of the field from the output.
        """
        for idx, stream in enumerate(self._streams):
            self._current_stream_index = idx
            mapping = self._field_mappings[idx]

            if mapping["fields"] is None or field_name in mapping["fields"]:
                async for output in stream.stream_output(
                    debounce_by=debounce_by
                ):
                    if mapping["update_output"]:
                        temp_result = PydanticAIAgentResult(output=output)
                        self._builder.update_from_pydantic_ai_result(
                            temp_result, fields=mapping["fields"]
                        )

                    if isinstance(output, BaseModel) and hasattr(
                        output, field_name
                    ):
                        yield getattr(output, field_name)
                    else:
                        yield None
            else:
                async for _ in stream.stream_output(debounce_by=debounce_by):
                    pass

            final_output = await stream.get_output()
            self._raw.append(PydanticAIAgentResult(output=final_output))

        self._is_complete = True

    async def finish_async(self) -> Result[Output]:
        """
        Finish the stream and return the final result.
        """
        if self._is_complete:
            return Result(
                output=self._builder.finalize(exclude_none=self._exclude_none),
                raw=self._raw,
            )

        for idx in range(self._current_stream_index, len(self._streams)):
            stream = self._streams[idx]
            mapping = self._field_mappings[idx]
            self._current_stream_index = idx

            final_output = await stream.get_output()

            if mapping["update_output"]:
                result = PydanticAIAgentResult(output=final_output)
                self._builder.update_from_pydantic_ai_result(
                    result, fields=mapping["fields"]
                )
                self._raw.append(result)

        self._is_complete = True

        return Result(
            output=self._builder.finalize(exclude_none=self._exclude_none),
            raw=self._raw,
        )

    def text(
        self, *, delta: bool = False, debounce_by: float | None = 0.1
    ) -> Iterator[str]:
        """
        Synchronously stream the text output.

        Args:
            delta: Whether to stream the delta of the text output.
            debounce_by: The debounce interval in seconds.

        Returns:
            The text output of the stream.
        """
        return _sync_async_iterator(
            self.stream_text(delta=delta, debounce_by=debounce_by)
        )

    def partial(self, *, debounce_by: float | None = 0.1) -> Iterator[Output]:
        """
        Stream the composition of the output as it is generated.

        Args:
            debounce_by: The debounce interval in seconds.

        Returns:
            The partial output of the stream.
        """
        return _sync_async_iterator(
            self.stream_partial(debounce_by=debounce_by)
        )

    def field(
        self, field_name: str, *, debounce_by: float | None = 0.1
    ) -> Iterator[Any]:
        """
        Synchronously stream a specific field from the output.

        Args:
            field_name: The name of the field to stream.
            debounce_by: The debounce interval in seconds.
        """
        return _sync_async_iterator(
            self.stream_field(field_name=field_name, debounce_by=debounce_by)
        )

    def finish(self) -> Result[Output]:
        """
        Finish the stream and return the final result.
        """
        loop = _get_event_loop()
        return loop.run_until_complete(self.finish_async())

    @property
    def is_streaming(self) -> bool:
        """
        Whether the stream is still in progress.
        """
        return not self._is_complete

    @property
    def result(self) -> Result[Output]:
        """
        The final result of the stream.
        """
        if not self._is_complete:
            raise RuntimeError(
                "Stream not yet complete - use finish() or iterate through the stream first"
            )
        if not self._raw:
            raise RuntimeError("Stream completed but no output was produced")
        return Result(
            output=self._builder.finalize(exclude_none=self._exclude_none),
            raw=self._raw,
        )

    @property
    def usage(self) -> PydanticAIUsage:
        """
        The usage of the stream.
        """
        total = PydanticAIUsage()
        for run in self._raw:
            total.incr(run.usage())
        return total

    def __repr__(self) -> str:
        return (
            f"Stream({type(self._builder.normalized).__name__}):\n"
            f" >>> Is Streaming: {self.is_streaming}\n"
            f" >>> Is Complete: {self._is_complete}\n"
        )

    def __rich__(self):
        from rich.console import RenderableType, Group
        from rich.rule import Rule
        from rich.text import Text

        renderables: list[RenderableType] = []

        renderables.append(
            Rule(title="✨ Stream", style="rule.line", align="left")
        )

        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Type: {type(self._builder.normalized).__name__}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Is Streaming: {self.is_streaming}[/dim italic]"
            )
        )
        renderables.append(
            Text.from_markup(
                f"[sandy_brown]>>>[/sandy_brown] [dim italic]Is Complete: {self._is_complete}[/dim italic]"
            )
        )

        if self._streams and len(self._streams) > 0:
            if (
                self._streams[0].response
                and self._streams[0].response.model_name
            ):
                renderables.append(
                    Text.from_markup(
                        f"[sandy_brown]>>>[/sandy_brown] [dim italic]Model: {self._streams[0].response.model_name}[/dim italic]"
                    )
                )

        return Group(*renderables)


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create event loop for sync operations.

    Matches the pattern used by pydantic_ai and pydantic_graph internals:
    uses `get_event_loop()` (not `get_running_loop()`) so the same loop
    is reused across sequential `run_until_complete` calls.
    """
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop


def _sync_async_iterator(
    async_iter: AsyncIterator[Output],
) -> Iterator[Output]:
    """Convert an async iterator to a sync iterator with true streaming.

    Unlike collecting all items first, this yields each item as it becomes
    available, providing true streaming behavior in sync contexts.
    """
    loop = _get_event_loop()
    while True:
        try:
            yield loop.run_until_complete(anext(async_iter))
        except StopAsyncIteration:
            break
