"""zyx.stream"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
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

__all__ = ("Stream",)


Output = TypeVar("Output")


@dataclass
class StreamFieldMapping:
    stream_index: int
    fields: List[str] | None
    update_output: bool


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
    _streams: List[PydanticAIAgentStream | None] = field(repr=False)
    _field_mappings: List[StreamFieldMapping] = field(repr=False)
    _stream_contexts: List[Any] = field(repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(
        default=None, init=False, repr=False
    )
    _runner: asyncio.Runner | None = field(
        default=None, init=False, repr=False
    )
    _loop_owner_thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )

    _current_stream_index: int = field(default=0, init=False, repr=False)
    _raw: List[PydanticAIAgentResult[Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _is_complete: bool = field(default=False, init=False, repr=False)
    _exclude_none: bool = field(default=False, repr=False)
    """When ``True``, ``finalize()`` will omit ``None``-valued fields from
    the result (useful for selective edits where only changed fields should
    appear)."""

    async def __aenter__(self) -> Stream[Output]:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._is_complete:
            await self.finish_async()

        for idx, stream_ctx in enumerate(self._stream_contexts):
            if stream_ctx is None:
                continue
            try:
                await stream_ctx.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
            self._stream_contexts[idx] = None
            self._streams[idx] = None

    async def _ensure_stream(self, idx: int) -> PydanticAIAgentStream:
        stream = self._streams[idx]
        if stream is not None:
            return stream
        stream_ctx = self._stream_contexts[idx]
        if stream_ctx is None:
            raise RuntimeError("Stream context was already closed")
        stream = await stream_ctx.__aenter__()
        self._streams[idx] = stream
        return stream

    async def _close_stream(self, idx: int) -> None:
        stream_ctx = self._stream_contexts[idx]
        if stream_ctx is None:
            return
        try:
            await stream_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        self._stream_contexts[idx] = None
        self._streams[idx] = None

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
        for idx, _ in enumerate(self._streams):
            self._current_stream_index = idx
            mapping = self._field_mappings[idx]

            stream = await self._ensure_stream(idx)
            try:
                text_iter = stream.stream_text(
                    delta=delta, debounce_by=debounce_by
                )
                try:
                    async for text_chunk in text_iter:
                        yield text_chunk
                finally:
                    aclose = getattr(text_iter, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception:
                            pass

                final_output = await _finalize_stream_output(stream)
                result = PydanticAIAgentResult(output=final_output)

                if mapping.update_output:
                    self._builder.update_from_pydantic_ai_result(
                        result, fields=mapping.fields
                    )

                self._raw.append(result)
            finally:
                await self._close_stream(idx)

        self._is_complete = True

    async def stream_partial(
        self, *, delta: bool = False, debounce_by: float | None = 0.1
    ) -> AsyncIterator[Output]:
        """
        Stream real-time updates of the target output. This method aggregates all runs
        taking place within the semantic operation, and is able to provide updates for
        multiple fields within the output.

        Args:
            delta: Whether to stream only the delta (changes) since the last yield.
                If ``False`` (default), yields the full accumulated output each time.
                If ``True``, yields only the changes since the last yield.
            debounce_by: The debounce interval in seconds.

        Yields:
            The partial output of the stream. If ``delta=True``, yields only the changes
            since the last yield. If ``delta=False``, yields the full accumulated output.
        """
        previous_output: Output | None = None

        for idx, _ in enumerate(self._streams):
            self._current_stream_index = idx
            mapping = self._field_mappings[idx]

            stream = await self._ensure_stream(idx)
            try:
                if mapping.update_output:
                    out_iter = self._builder.update_from_pydantic_ai_stream(
                        stream=stream,
                        fields=mapping.fields,
                        debounce_by=debounce_by,
                    )
                    try:
                        async for partial in out_iter:
                            if delta:
                                if previous_output is not None:
                                    # Calculate delta by comparing with previous output
                                    if isinstance(partial, str) and isinstance(
                                        previous_output, str
                                    ):
                                        # For strings, yield only the new characters
                                        if len(partial) > len(previous_output):
                                            delta_str = partial[
                                                len(previous_output) :
                                            ]
                                            yield delta_str
                                            previous_output = partial  # Update after yielding delta
                                        elif partial != previous_output:
                                            # String changed but got shorter (shouldn't happen normally)
                                            yield partial
                                            previous_output = partial
                                    elif isinstance(
                                        partial, BaseModel
                                    ) and isinstance(
                                        previous_output, BaseModel
                                    ):
                                        # For BaseModel, create a delta with only changed fields
                                        delta_dict = {}
                                        for (
                                            field_name
                                        ) in partial.model_fields.keys():
                                            prev_val = getattr(
                                                previous_output,
                                                field_name,
                                                None,
                                            )
                                            curr_val = getattr(
                                                partial, field_name, None
                                            )
                                            if prev_val != curr_val:
                                                delta_dict[field_name] = (
                                                    curr_val
                                                )

                                        if delta_dict:
                                            # Create a partial model with only changed fields
                                            delta_output = self._builder.partial_type.model_validate(
                                                delta_dict
                                            )
                                            yield delta_output
                                            previous_output = partial  # Update after yielding delta
                                    else:
                                        # For other simple types, only yield if value changed
                                        if partial != previous_output:
                                            yield partial
                                            previous_output = partial  # Update after yielding
                                else:
                                    # First iteration - yield full output
                                    yield partial
                                    previous_output = partial
                            else:
                                # Not delta mode - always yield full accumulated output
                                yield partial

                        final_output = await _finalize_stream_output(stream)
                        self._raw.append(
                            PydanticAIAgentResult(output=final_output)
                        )
                    finally:
                        aclose = getattr(out_iter, "aclose", None)
                        if aclose is not None:
                            try:
                                await aclose()
                            except Exception:
                                pass
                else:
                    out_iter = stream.stream_output(debounce_by=debounce_by)
                    try:
                        async for _ in out_iter:
                            pass

                        final_output = await _finalize_stream_output(stream)
                        self._raw.append(
                            PydanticAIAgentResult(output=final_output)
                        )
                    finally:
                        aclose = getattr(out_iter, "aclose", None)
                        if aclose is not None:
                            try:
                                await aclose()
                            except Exception:
                                pass
            finally:
                await self._close_stream(idx)

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
        for idx, _ in enumerate(self._streams):
            self._current_stream_index = idx
            mapping = self._field_mappings[idx]

            stream = await self._ensure_stream(idx)
            try:
                if mapping.fields is None or field_name in mapping.fields:
                    out_iter = stream.stream_output(debounce_by=debounce_by)
                    try:
                        async for output in out_iter:
                            if mapping.update_output:
                                temp_result = PydanticAIAgentResult(
                                    output=output
                                )
                                self._builder.update_from_pydantic_ai_result(
                                    temp_result, fields=mapping.fields
                                )

                            if isinstance(output, BaseModel) and hasattr(
                                output, field_name
                            ):
                                yield getattr(output, field_name)
                            else:
                                yield None

                        final_output = await _finalize_stream_output(stream)
                        self._raw.append(
                            PydanticAIAgentResult(output=final_output)
                        )
                    finally:
                        aclose = getattr(out_iter, "aclose", None)
                        if aclose is not None:
                            try:
                                await aclose()
                            except Exception:
                                pass
                else:
                    out_iter = stream.stream_output(debounce_by=debounce_by)
                    try:
                        async for _ in out_iter:
                            pass

                        final_output = await _finalize_stream_output(stream)
                        self._raw.append(
                            PydanticAIAgentResult(output=final_output)
                        )
                    finally:
                        aclose = getattr(out_iter, "aclose", None)
                        if aclose is not None:
                            try:
                                await aclose()
                            except Exception:
                                pass
            finally:
                await self._close_stream(idx)

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
            mapping = self._field_mappings[idx]
            self._current_stream_index = idx
            stream = await self._ensure_stream(idx)
            try:
                final_output = await _finalize_stream_output(stream)

                if mapping.update_output:
                    result = PydanticAIAgentResult(output=final_output)
                    self._builder.update_from_pydantic_ai_result(
                        result, fields=mapping.fields
                    )
                    self._raw.append(result)
            finally:
                await self._close_stream(idx)

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
            self.stream_text(delta=delta, debounce_by=debounce_by),
            loop=self._loop,
            loop_owner_thread=self._loop_owner_thread,
            on_exhausted=lambda: _stop_loop_if_remote(self),
        )

    def partial(
        self, *, delta: bool = False, debounce_by: float | None = 0.1
    ) -> Iterator[Output]:
        """
        Stream the composition of the output as it is generated.

        Args:
            delta: Whether to stream only the delta (changes) since the last yield.
                If ``False`` (default), yields the full accumulated output each time.
            debounce_by: The debounce interval in seconds.

        Returns:
            The partial output of the stream. If ``delta=True``, yields only the changes
            since the last yield. If ``delta=False``, yields the full accumulated output.
        """
        return _sync_async_iterator(
            self.stream_partial(delta=delta, debounce_by=debounce_by),
            loop=self._loop,
            loop_owner_thread=self._loop_owner_thread,
            on_exhausted=lambda: _stop_loop_if_remote(self),
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
            self.stream_field(field_name=field_name, debounce_by=debounce_by),
            loop=self._loop,
            loop_owner_thread=self._loop_owner_thread,
            on_exhausted=lambda: _stop_loop_if_remote(self),
        )

    def finish(self) -> Result[Output]:
        """
        Finish the stream and return the final result.
        """
        loop = self._loop or _get_event_loop()
        use_remote = (
            self._loop_owner_thread is not None
            and threading.current_thread() is not self._loop_owner_thread
        )
        if use_remote and loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self.finish_async(), loop
            )
            result = future.result()
            _stop_loop_if_remote(self)
        else:
            result = loop.run_until_complete(self.finish_async())
            if self._runner is not None:
                self._runner.close()
                self._runner = None
                self._loop = None
        return result

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
            first_stream = self._streams[0]
            if (
                first_stream is not None
                and first_stream.response
                and first_stream.response.model_name
            ):
                renderables.append(
                    Text.from_markup(
                        f"[sandy_brown]>>>[/sandy_brown] [dim italic]Model: {first_stream.response.model_name}[/dim italic]"
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
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    loop_owner_thread: threading.Thread | None = None,
    on_exhausted: Any = None,
) -> Iterator[Output]:
    """Convert an async iterator to a sync iterator with true streaming.

    When loop runs in another thread (loop_owner_thread set and current thread
    is not it), runs the async iteration on that loop and yields from a queue.
    """
    loop = loop or _get_event_loop()
    use_remote = (
        loop_owner_thread is not None
        and threading.current_thread() is not loop_owner_thread
    )
    if use_remote:
        q: queue.Queue[Output | None] = queue.Queue()

        async def consume() -> None:
            try:
                async for x in async_iter:
                    q.put(x)
            finally:
                aclose = getattr(async_iter, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
                q.put(None)

        future = asyncio.run_coroutine_threadsafe(consume(), loop)
        try:
            while True:
                x = q.get()
                if x is None:
                    break
                yield x
        finally:
            if not future.done():
                future.cancel()
            try:
                future.result()
            except Exception:
                pass
            if on_exhausted is not None:
                on_exhausted()
        return

    try:
        while True:
            try:
                yield loop.run_until_complete(anext(async_iter))
            except StopAsyncIteration:
                break
    finally:
        aclose = getattr(async_iter, "aclose", None)
        if aclose is not None:
            try:
                loop.run_until_complete(aclose())
            except Exception:
                pass


def _stop_loop_if_remote(stream: Stream[Any]) -> None:
    """Stop or close the loop when the stream owns it."""
    if stream._loop is not None and stream._loop_owner_thread is not None:
        # Wait for all pending tasks to complete before stopping
        async def _wait_and_stop():
            # Give a brief moment for any pending cleanup operations
            await asyncio.sleep(0.05)

            # Check if there are any pending tasks and wait for them
            current = asyncio.current_task()
            tasks = [
                t
                for t in asyncio.all_tasks(stream._loop)
                if not t.done() and t is not current
            ]
            if tasks:
                await asyncio.wait(tasks, timeout=0.5)

        future = asyncio.run_coroutine_threadsafe(
            _wait_and_stop(), stream._loop
        )
        try:
            future.result(timeout=1.0)
        except Exception:
            pass

        stream._loop.call_soon_threadsafe(stream._loop.stop)
        stream._loop = None
        stream._loop_owner_thread = None
        return
    if stream._runner is not None:
        stream._runner.close()
        stream._runner = None
        stream._loop = None


async def _finalize_stream_output(
    stream: PydanticAIAgentStream,
) -> Any:
    """Finalize output without re-iterating raw streams when already complete."""
    if getattr(stream, "is_complete", False):
        return await stream.validate_response_output(stream.response)
    return await stream.get_output()
