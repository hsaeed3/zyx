__all__ = [
    "completion",

    "llm",
    "data",

    "ChatClient",

    "Agents",
    "classify",
    "code",
    "create_system_prompt",
    "extract",
    "function",
    "generate",
    "least_to_most",
    "optimize_system_prompt",
    "plan",
    "self_consistency",
    "self_refine",
    "step_back",
    "tree_of_thought",
    "query",

    "SqlStore",
    "Rag",
    "VectorStore",

    "logger",
    "utils",

    "zyx"
]


from .lib.utils._loader import loader


from .lib.routes import (
    llm as llm,
    data as data
)


class completion(loader):
    pass


completion.init("zyx.lib.client.chat", "completion")


from .lib.routes.data import (
    sql_store as SqlStore,
    rag as Rag,
    vector_store as VectorStore
)


from .lib.routes.llm import (
    ChatClient,
    agents as Agents,
    classify,
    code,
    create_system_prompt,
    extract,
    function,
    generate,
    least_to_most,
    optimize_system_prompt,
    plan,
    self_consistency,
    self_refine,
    step_back,
    tree_of_thought,
    query
)


class logger(loader):
    pass


logger.init("loguru", "logger")


from .lib import utils


from pydantic import BaseModel
from typing import Union, List, Optional, Literal, Type


class ModuleWrapper:
    def __init__(self, module):
        self._module = module
        self._chat_client = None

    def __call__(self, 
                 messages: Union[str, List[dict]],
                 model: str = "gpt-4o-mini",
                 client: Literal["openai", "litellm"] = "openai",
                 response_model: Optional[Type[BaseModel]] = None,
                 mode: Optional[str] = "tool_call",
                 max_retries: Optional[int] = 3,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 organization: Optional[str] = None,
                 run_tools: Optional[bool] = True,
                 tools: Optional[List] = None,
                 parallel_tool_calls: Optional[bool] = False,
                 tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 stop: Optional[List[str]] = None,
                 stream: Optional[bool] = False,
                 verbose: Optional[bool] = False):
        return completion(
            messages=messages,
            model=model,
            client=client,
            response_model=response_model,
            mode=mode,
            max_retries=max_retries,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            run_tools=run_tools,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
            verbose=verbose
        )
    
    def __getattr__(self, name):
        return getattr(self._module, name)

# Wrap the current module
import sys
sys.modules[__name__] = ModuleWrapper(sys.modules[__name__])







