"""
Base types for zyx package.
Contains core type definitions to avoid circular dependencies.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pydantic import BaseModel as PydanticBaseModel

# Re-export pydantic BaseModel to avoid direct dependency
BaseModel = PydanticBaseModel

class Message(BaseModel):
    """Base message type for completions."""
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class Tool(BaseModel):
    """Base tool type for function calling."""
    name: Optional[str] = None
    function: Optional[Callable] = None
    arguments: Optional[Dict[str, Any]] = None
    formatted_function: Optional[Dict[str, Any]] = None

    def _execute(self, verbose: bool = False, **kwargs) -> Any:
        """Execute the tool function with given arguments."""
        if verbose:
            print(f"Executing tool: {self.name}")
            print(f"Arguments: {kwargs}")
        return self.function(**kwargs)

# Type aliases
ToolType = Union[str, Callable, Tool]
ChatModel = str  # Model identifier string
Client = Type["Client"]  # Forward reference for Client type

class Choice(BaseModel):
    """Completion choice containing message and other metadata."""
    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None

class Usage(BaseModel):
    """Token usage information."""
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class Completion(BaseModel):
    """Base completion response type."""
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: str
    usage: Usage
    system_fingerprint: Optional[str] = None

# Constants
DEFAULT_TIMEOUT = 600  # 10 minutes
DEFAULT_MAX_RETRIES = 2
ZYX_DEFAULT_MODEL = "gpt-4"

# Enums and Literals can be defined here as well
from enum import Enum
class InstructorMode(str, Enum):
    """Instructor response modes."""
    TOOL_CALL = "tool_call"
    JSON = "json"
    MD_JSON = "md_json"
    TOOLS = "tools"