import enum
from pydantic import BaseModel
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)


## -- Instructor Mode -- ##
## This was directly ported from instructor
## https://github.com/jxnl/instructor/
class Mode(enum.Enum):
    """The mode to use for patching the client"""

    FUNCTIONS = "function_call"
    PARALLEL_TOOLS = "parallel_tool_call"
    TOOLS = "tool_call"
    MISTRAL_TOOLS = "mistral_tools"
    JSON = "json_mode"
    JSON_O1 = "json_o1"
    MD_JSON = "markdown_json_mode"
    JSON_SCHEMA = "json_schema_mode"
    ANTHROPIC_TOOLS = "anthropic_tools"
    ANTHROPIC_JSON = "anthropic_json"
    COHERE_TOOLS = "cohere_tools"
    VERTEXAI_TOOLS = "vertexai_tools"
    VERTEXAI_JSON = "vertexai_json"
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"
    COHERE_JSON_SCHEMA = "json_object"
    TOOLS_STRICT = "tools_strict"


InstructorMode = Literal[
    "function_call", "parallel_tool_call", "tool_call", "mistral_tools",
    "json_mode", "json_o1", "markdown_json_mode", "json_schema_mode",
    "anthropic_tools", "anthropic_json", "cohere_tools", "vertexai_tools",
    "vertexai_json", "gemini_json", "gemini_tools", "json_object",
    "tools_strict"
]


def get_mode(mode : InstructorMode) -> Mode:
    return Mode(mode)


## -- Tools -- ##


ToolType = Union[Dict[str, Any], Type[BaseModel], Callable]


class Tool(BaseModel):
    name : Optional[str]
    tool : ToolType
    openai_tool : Optional[Dict[str, Any]]


class ToolResponse(BaseModel):
    name : Any
    args : Any
    output : Optional[Any]


## -- Client Config -- ##


class ClientConfig(BaseModel):
    client : Literal["openai", "litellm"]
    api_key : Optional[str]
    base_url : Optional[str]
    organization : Optional[str]
    verbose : bool


class ClientProviders(BaseModel):
    client : Any
    instructor : Any


class Client(BaseModel):
    config : ClientConfig


## -- Completion Types -- ##


class CompletionArgs(BaseModel):
    messages : List[Dict[str, str]]
    model : str
    response_model : Optional[Type[BaseModel]]
    tools : Optional[List[Dict[str, Any]]]
    parallel_tool_calls : Optional[bool]
    tool_choice : Optional[Literal["none", "auto", "required"]]
    max_tokens : Optional[int]
    temperature : Optional[float]
    top_p : Optional[float]
    frequency_penalty : Optional[float]
    presence_penalty : Optional[float]
    stop : Optional[List[str]]
    stream : Optional[bool]


CompletionResponse = Union[
    Type[BaseModel],
    Dict[str, str],
    Generator
]
    