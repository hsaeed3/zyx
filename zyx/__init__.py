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
    "utils"
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


