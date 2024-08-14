__all__ = ["ClientModeParams", "ClientParams"]

# --- zyx ----------------------------------------------------------------

from ..core.ext import BaseModel
from typing import Any, Callable, List, Literal, Optional, Union

# --- LLM Completion Client ---

ClientModeParams = Literal["json", "json_schema", "md_json", "parallel", "tools"]


class ClientToolParams(BaseModel):
    tools: List[Union[Callable, dict, BaseModel]] = None
    openai_tools: List[dict] = None
    mapping: Optional[dict] = None


class ClientParams(BaseModel):
    messages: Union[str, list[dict]] = None

    model: Optional[str] = "gpt-4o-mini"
    tools: Optional[ClientToolParams] = None
    run_tools: Optional[bool] = True
    response_model: Optional[Union[Any, BaseModel]] = None
    mode: Optional[ClientModeParams] = "tools"

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    organization: Optional[str] = None

    top_p: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 3
    max_retries: Optional[int] = None
    kwargs: Optional[dict] = None
