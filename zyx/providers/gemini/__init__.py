"""zyx.providers.gemini"""

from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    List,
)

from .. import ProviderUtils

__all__ = ("GEMINI_PROVIDER_UTILS",)


def handle_request_params_gemini(
    messages: List[Any],
    tools: List[Dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Process request parameters for Gemini/GenAI/VertexAI clients.
    Returns kwargs to pass to the API.
    """
    params = kwargs.copy()
    params["messages"] = messages

    if tools:
        params["tools"] = tools

    return params


def parse_raw_response_gemini(
    raw_response: Any,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Parses a raw response or response chunk from Gemini/GenAI/VertexAI
    API and returns a dictionary with the following keys:

    - content : str | None (the text content of the response, if any)
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

    # Try to get text directly (GenAI style)
    text = getattr(raw_response, "text", None)
    if text is not None:
        result["content"] = text

    # Extract from candidates (Gemini/VertexAI style)
    candidates = getattr(raw_response, "candidates", None)
    if candidates:
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if content:
            parts = getattr(content, "parts", None)
            if parts:
                text_parts = []
                tool_calls = []

                for part in parts:
                    # Check for text
                    part_text = getattr(part, "text", None)
                    if part_text:
                        text_parts.append(part_text)

                    # Check for function call
                    fc = getattr(part, "function_call", None)
                    if fc:
                        tool_calls.append(
                            {
                                "id": getattr(fc, "id", fc.name),
                                "name": fc.name,
                                "arguments": dict(fc.args)
                                if fc.args
                                else {},
                            }
                        )

                if text_parts and result["content"] is None:
                    result["content"] = "".join(text_parts)

                if tool_calls:
                    result["tool_calls"] = tool_calls

        # Extract finish reason
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            result["finish_reason"] = str(finish_reason)

    # Extract usage (usage_metadata for Gemini)
    usage_metadata = getattr(raw_response, "usage_metadata", None)
    if usage_metadata:
        result["usage"] = {
            "prompt_tokens": getattr(
                usage_metadata, "prompt_token_count", 0
            )
            or 0,
            "completion_tokens": getattr(
                usage_metadata, "candidates_token_count", 0
            )
            or 0,
            "total_tokens": getattr(usage_metadata, "total_token_count", 0)
            or 0,
        }

    return result


def format_tool_result_gemini(
    tool_call_id: str, name: str, result: Any, **kwargs: Any
) -> Dict[str, Any] | Any:
    """
    Formats a tool result for Gemini/GenAI/VertexAI APIs.
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

    # Check for specific variant
    variant = kwargs.get("variant", "gemini")

    if variant == "genai":
        try:
            from google.genai import types

            return types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=name, response={"result": content}
                    )
                ],
            )
        except ImportError:
            pass

    if variant == "vertexai":
        try:
            import vertexai.generative_models as gm

            return gm.Content(
                parts=[
                    gm.Part.from_function_response(
                        name=name, response={"result": content}
                    )
                ]
            )
        except ImportError:
            pass

    # Default Gemini format
    try:
        from google.genai import types

        return {
            "role": "function",
            "parts": [
                types.Part(
                    function_response=types.FunctionResponse(
                        name=name,
                        response={"result": content},
                    )
                )
            ],
        }
    except ImportError:
        # Fallback if google.genai not available
        return {
            "role": "function",
            "parts": [
                {
                    "function_response": {
                        "name": name,
                        "response": {"result": content},
                    }
                }
            ],
        }


GEMINI_PROVIDER_UTILS: ProviderUtils = {
    "handle_request_params": handle_request_params_gemini,
    "parse_raw_response": parse_raw_response_gemini,
    "format_tool_result": format_tool_result_gemini,
}
