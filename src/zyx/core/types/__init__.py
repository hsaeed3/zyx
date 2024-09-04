__all__ = ["ClientModeParams", "ClientParams"]

# --- zyx ----------------------------------------------------------------

from ..main import BaseModel, Field
from typing import Any, Callable, List, Literal, Optional, Union, Dict
from uuid import uuid4

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


# --- Agents ---


class TaskIntent(BaseModel):
    intent: str
    description: str
    priority: int = Field(default=1, ge=1, le=5)


class TaskDelegation(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    intent: TaskIntent
    assigned_worker: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None


class SupervisorResponse(BaseModel):
    message: Optional[str] = None
    delegations: Optional[List[TaskDelegation]] = None


class WorkerResponse(BaseModel):
    task_id: str
    status: str
    result: Any
    message: Optional[str] = None


class AgentParams(BaseModel):
    agent_id: str
    agent_type: str
    tools: List[Any] = []
    instructions: Optional[str] = None
    completion_params: Optional[ClientParams] = None
