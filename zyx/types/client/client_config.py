"""
zyx.types.client.client_config

Contains the client `config` type, which is used to configure the client.
"""

from __future__ import annotations

# [Imports]
import httpx
from typing import Optional, Union, Mapping
from pydantic import BaseModel, ConfigDict


# ===================================================================
# [Client Config Type]
# ===================================================================


class ClientConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # [Shared Params]
    api_key: Optional[str] = None
    """The API key or `api_base` to use for the primary client endpoint."""

    base_url: Optional[str] = None
    """The base URL to use for the primary client endpoint."""

    organization: Optional[str] = None
    """The organization to use for the primary client endpoint."""

    timeout: Optional[Union[float, httpx.Timeout, None]] = None
    """The timeout to use for the primary client endpoint."""

    max_retries: Optional[int] = None
    """The maximum number of retries to use for the primary client endpoint."""

    default_headers: Optional[Mapping[str, str]] = None
    """The default headers to use for the primary client endpoint."""

    # [OpenAI Specific Params]
    project: Optional[str] = None
    """The project to use for the primary client endpoint."""

    default_query: Optional[Mapping[str, object]] = None
    """The default query parameters to use for the primary client endpoint."""

    websocket_base_url: Optional[str] = None
    """The websocket base URL to use for the primary client endpoint."""

    http_client: Optional[httpx.Client] = None
    """The HTTP client to use for the primary client endpoint."""

    _strict_response_validation: Optional[bool] = False
    """Whether to validate the response against the schema."""

    # ===================================================================
    # [Helper Functions]
    # ===================================================================

    def dump_for_openai(self) -> dict:
        """Dumps the config for the OpenAI client."""

        return self.model_dump()

    def dump_for_litellm(self) -> dict:
        """Dumps the config for the LiteLLM client."""

        excluded_keys = [
            "project",
            "default_query",
            "websocket_base_url",
            "http_client",
            "_strict_response_validation",
        ]

        return {k: v for k, v in self.model_dump().items() if k not in excluded_keys}
