"""zyx.providers.openai"""

from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    List,
)

from .. import ProviderUtils

__all__ = ("OPENAI_PROVIDER_UTILS",)


def handle_request_params_openai(
    messages: List[Any],
    tools: List[Dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Process request parameters for OpenAI-compatible clients.
    Returns kwargs to pass to the API.
    """
    params = kwargs.copy()
    params["messages"] = messages

    if tools:
        params["tools"] = tools

    return params


def parse_raw_response_openai(
    raw_response: Any,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Parses a raw response or response chunk from the OpenAI
    API and returns a dictionary with the following keys:

    - content : str | None (the text content of the response, if
        any)
    - tool_calls : List[Dict[str, Any]] | None (the tool calls made
        during the response, if any)
    - usage : Dict[str, int] | None (the usage statistics for the
        response, if any)
    - finish_reason : str | None (the reason the response finished,
        if any)
    """
    result: Dict[str, Any] = {
        "content": None,
        "tool_calls": None,
        "usage": None,
        "finish_reason": None,
    }

    if raw_response is None:
        return result

    choices = getattr(raw_response, "choices", None)
    if not choices:
        return result

    if stream:
        # Parse streaming chunk
        delta = getattr(choices[0], "delta", None)
        if delta:
            content = getattr(delta, "content", None)
            if content:
                result["content"] = content

            # Extract tool call deltas
            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                parsed_calls = []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    parsed_calls.append(
                        {
                            "id": getattr(tc, "id", None) or "",
                            "name": getattr(fn, "name", None) or ""
                            if fn
                            else "",
                            "arguments": getattr(fn, "arguments", None)
                            or ""
                            if fn
                            else "",
                        }
                    )
                result["tool_calls"] = parsed_calls

        # Extract finish reason (usually in last chunk)
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason:
            result["finish_reason"] = finish_reason
    else:
        # Parse complete response
        message = choices[0].message
        content = getattr(message, "content", None)
        if content:
            result["content"] = content

        # Extract tool calls
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            parsed_calls = []
            for tc in tool_calls:
                parsed_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )
            result["tool_calls"] = parsed_calls

        # Extract finish reason
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason:
            result["finish_reason"] = finish_reason

    # Extract usage (present in complete response or final streaming chunk)
    usage = getattr(raw_response, "usage", None)
    if usage:
        result["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0)
            or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    return result


def format_tool_result_openai(
    tool_call_id: str, name: str, result: Any, **kwargs: Any
) -> Dict[str, Any]:
    """
    Formats a tool result for the OpenAI API.
    """
    # Serialize result
    if isinstance(result, Exception):
        content = f"Error: {type(result).__name__}: {result}"
    elif isinstance(result, str):
        content = result
    else:
        try:
            content = json.dumps(result, default=str)
        except (TypeError, ValueError):
            content = str(result)

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


OPENAI_PROVIDER_UTILS: ProviderUtils = {
    "handle_request_params": handle_request_params_openai,
    "parse_raw_response": parse_raw_response_openai,
    "format_tool_result": format_tool_result_openai,
}
