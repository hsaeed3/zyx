# zyx.resources.types.config.client
# client configuration types

from __future__ import annotations

__all__ = [
    "ClientConfig",
    "ClientChecks",
    "ClientProvider",
]


from openai._constants import httpx, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from pydantic import BaseModel, ConfigDict
from typing import Mapping, Literal, Optional, Union


# client provider
ClientProvider = Literal["openai", "litellm"]


# TODO
# client checks
# internal helper for client inits & resources
class ClientChecks(BaseModel):

    # arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # base client check
    is_client_initialized : bool = False

    # instructor patch check
    is_instructor_initialized : bool = False
    

# TODO
# client config
class ClientConfig(BaseModel):

    # arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # internal (provider)
    provider : Optional[ClientProvider] = None

    # base client args
    api_key : Optional[str] = None
    base_url : Optional[str] = None
    organization : Optional[str] = None
    project : Optional[str] = None
    timeout : Optional[Union[float, httpx.Timeout]] = DEFAULT_TIMEOUT
    max_retries : Optional[int] = DEFAULT_MAX_RETRIES
    default_headers : Optional[Mapping[str, str]] = None
    default_query : Optional[Mapping[str, object]] = None
    http_client : Optional[httpx.Client] = None

    # pruned httpx args (if needed (added for simplicity))
    verify_ssl : Optional[bool] = None
    http_args : Optional[Mapping[str, object]] = None

    # verbosity
    verbose : Optional[bool] = None
