"""zyx.providers.anthropic"""

from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    List,
)

from .. import ProviderUtils

__all__ = ("ANTHROPIC_PROVIDER_UTILS",)


def handle_request_params_anthropic(
    messages: List[Any],
    tools: List[Dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Process request parameters for Anthropic clients.
    Returns kwargs to pass to the API.
    """
    params = kwargs.copy()
    params["messages"] = messages

    if tools:
        params["tools"] = tools

    return params


def parse_raw_response_anthropic(
    raw_response: Any,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Parses a raw response or response chunk from the Anthropic
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

    if stream:
        # Parse streaming chunk
        chunk_type = getattr(raw_response, "type", None)

        # Handle content_block_start (new tool use or text block)
        if chunk_type == "content_block_start":
            block = getattr(raw_response, "content_block", None)
            if block:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    # Tool use starting
                    result["tool_calls"] = [
                        {
                            "id": block.id,
                            "name": block.name,
                            "arguments": "",
                        }
                    ]

        # Handle content_block_delta (text or tool input delta)
        elif chunk_type == "content_block_delta":
            delta = getattr(raw_response, "delta", None)
            if delta:
                delta_type = getattr(delta, "type", None)

                if delta_type == "text_delta":
                    text = getattr(delta, "text", None)
                    if text:
                        result["content"] = text

                elif delta_type == "input_json_delta":
                    # Tool input delta
                    partial_json = getattr(delta, "partial_json", None)
                    if partial_json:
                        result["tool_calls"] = [
                            {
                                "id": "",
                                "name": "",
                                "arguments": partial_json,
                            }
                        ]

        # Handle message_delta (usage and stop_reason updates)
        elif chunk_type == "message_delta":
            delta = getattr(raw_response, "delta", None)
            if delta:
                stop_reason = getattr(delta, "stop_reason", None)
                if stop_reason:
                    result["finish_reason"] = stop_reason

            usage = getattr(raw_response, "usage", None)
            if usage:
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                result["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": output_tokens,
                    "total_tokens": output_tokens,
                }
    else:
        # Parse complete response
        content_blocks = getattr(raw_response, "content", None)
        if content_blocks and isinstance(content_blocks, list):
            text_parts = []
            tool_calls = []

            for block in content_blocks:
                block_type = getattr(block, "type", None)

                if block_type == "text":
                    text = getattr(block, "text", None)
                    if text:
                        text_parts.append(text)

                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "arguments": block.input
                            if isinstance(block.input, dict)
                            else {},
                        }
                    )

            if text_parts:
                result["content"] = "".join(text_parts)

            if tool_calls:
                result["tool_calls"] = tool_calls

        # Extract finish reason (stop_reason in Anthropic)
        stop_reason = getattr(raw_response, "stop_reason", None)
        if stop_reason:
            result["finish_reason"] = stop_reason

        # Extract usage
        usage = getattr(raw_response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            result["usage"] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

    return result


def format_tool_result_anthropic(
    tool_call_id: str, name: str, result: Any, **kwargs: Any
) -> Dict[str, Any]:
    """
    Formats a tool result for the Anthropic API.
    """
    # Check for explicit error flag in kwargs
    is_error = kwargs.get("is_error", False)

    # Serialize result
    if isinstance(result, Exception):
        content = f"Error: {type(result).__name__}: {result}"
        is_error = True
    elif isinstance(result, str):
        content = result
    else:
        try:
            content = json.dumps(result, default=str)
        except (TypeError, ValueError):
            content = str(result)

    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content,
                "is_error": is_error,
            }
        ],
    }


ANTHROPIC_PROVIDER_UTILS: ProviderUtils = {
    "handle_request_params": handle_request_params_anthropic,
    "parse_raw_response": parse_raw_response_anthropic,
    "format_tool_result": format_tool_result_anthropic,
}
