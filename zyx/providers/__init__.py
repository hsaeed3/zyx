"""zyx.providers"""

from typing import Any, Callable, Dict, TypedDict


class ProviderUtils(TypedDict):
    """
    Provider-specific utilities used used for request construction
    and response parsing during invocations.
    """

    handle_request_params: Callable[[...], Dict[str, Any]]
    parse_raw_response: Callable[[Any, bool], Dict[str, Any]]
    format_tool_result: Callable[[str, str, Any], Dict[str, Any]]
