"""zyx.gen.make"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Type, TypeVar, TYPE_CHECKING

from ..core.context import Context, get_active_context, resolve_context
from ..model import Model, ModelSettings, ModelResponse
from ..result import Result

if TYPE_CHECKING:
    from instructor.core.client import (
        Instructor,
        AsyncInstructor,
    )
    from instructor.models import KnownModelName


T = TypeVar("T")


async def amake(
    context: List[Dict[str, Any] | Context | str] | Context | str,
    target: Type[T] | Type[str] = str,
    model: KnownModelName | Model | str = "openai/gpt-4o-mini",
    client: Instructor | AsyncInstructor | None = None,
    settings: ModelSettings | None = None,
    **kwargs: Any,
) -> Result[T]:
    """
    Asynchronously generate a response from the given context.

    Parameters
    ----------
    context : List[Dict | Context | str] | Context | str
        The input context. Can be a string (user message), a Context object,
        or a list mixing Context objects, dicts (raw messages), and strings.
    target : Type[T], optional
        The target type for structured output. Defaults to str.
    model : KnownModelName | Model | str, optional
        The model to use. Can be a model string, or an existing Model instance.
    client : Instructor | AsyncInstructor, optional
        An existing instructor client to use instead of creating one.
    settings : ModelSettings, optional
        Model settings to apply.
    **kwargs : Any
        Additional arguments passed to the model request.

    Returns
    -------
    Result[T]
        The generated result containing the output.
    """
    messages = resolve_context(context)

    if isinstance(model, Model):
        m = model
    else:
        m = Model(model=model, client=client, settings=settings)

    response: ModelResponse[T] = await m.arequest(
        messages=messages,
        target=target,
        stream=False,
        **kwargs,
    )

    # Update active context if present
    active_ctx = get_active_context()
    if active_ctx is not None:
        # Add user message(s) from the input
        if isinstance(context, str):
            active_ctx.user(context)
        elif isinstance(context, list):
            for item in context:
                if isinstance(item, str):
                    active_ctx.user(item)

        # Add assistant response
        if response.output is not None:
            if isinstance(response.output, str):
                content = response.output
            else:
                # For structured outputs, serialize to JSON for conversation history
                if hasattr(response.output, "model_dump_json"):
                    content = response.output.model_dump_json()
                elif hasattr(response.output, "model_dump"):
                    import json

                    content = json.dumps(response.output.model_dump())
                else:
                    content = str(response.output)
            active_ctx.assistant(content)

    return Result(output=response.output)


def make(
    context: List[Dict[str, Any] | Context | str] | Context | str,
    target: Type[T] | T | Type[str] = str,
    model: KnownModelName | Model | str = "openai/gpt-4o-mini",
    client: Instructor | AsyncInstructor | None = None,
    settings: ModelSettings | None = None,
    **kwargs: Any,
) -> Result[T]:
    """
    Synchronously generate a response from the given context.

    Parameters
    ----------
    context : List[Dict | Context | str] | Context | str
        The input context. Can be a string (user message), a Context object,
        or a list mixing Context objects, dicts (raw messages), and strings.
    target : Type[T], optional
        The target type for structured output. Defaults to str.
    model : KnownModelName | Model | str, optional
        The model to use. Can be a model string, or an existing Model instance.
    client : Instructor | AsyncInstructor, optional
        An existing instructor client to use instead of creating one.
    settings : ModelSettings, optional
        Model settings to apply.
    **kwargs : Any
        Additional arguments passed to the model request.

    Returns
    -------
    Result[T]
        The generated result containing the output.
    """
    return _run_sync(
        amake(context, target, model, client, settings, **kwargs)
    )


def _run_sync(coro: Any) -> Any:
    """Run coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)
    else:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
