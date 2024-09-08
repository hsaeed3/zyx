__all__ = [
    "Client",
    "classify",
    "code",
    "completion",
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
]


from ..utils.loader import Loader


class Client(Loader):
    pass


Client.init("zyx._client.completion", "CompletionClient")


class classify(Loader):
    pass


classify.init("zyx._client.llm.classify", "classify")


class completion(Loader):
    pass


completion.init("zyx._client.completion", "completion")


class code(Loader):
    pass


code.init("zyx._client.llm.code", "code")


class create_system_prompt(Loader):
    pass


create_system_prompt.init("zyx._client.llm.create_system_prompt", "create_system_prompt")


class extract(Loader):
    pass


extract.init("zyx._client.llm.extract", "extract")


class function(Loader):
    pass


function.init("zyx._client.llm.function", "function")


class generate(Loader):
    pass


generate.init("zyx._client.llm.generate", "generate")


class least_to_most(Loader):
    pass


least_to_most.init("zyx._client.llm.least_to_most", "least_to_most")


class optimize_system_prompt(Loader):
    pass


optimize_system_prompt.init("zyx._client.llm.optimize_system_prompt", "optimize_system_prompt")


class plan(Loader):
    pass


plan.init("zyx._client.llm.plan", "plan")


class self_consistency(Loader):
    pass


self_consistency.init("zyx._client.llm.self_consistency", "self_consistency")


class self_refine(Loader):
    pass


self_refine.init("zyx._client.llm.self_refine", "self_refine")


class step_back(Loader):
    pass


class tree_of_thought(Loader):
    pass


tree_of_thought.init("zyx._client.llm.tree_of_thought", "tree_of_thought")
