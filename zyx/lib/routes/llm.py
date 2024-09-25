__all__ = [
    "ChatClient",

    "agents",
    "completion",

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
    "query"
]


from ..utils._loader import loader


class ChatClient(loader):
    pass


ChatClient.init("zyx.lib.client.chat", "ChatClient")


class agents(loader):
    pass


agents.init("zyx.lib.client.resources.agents", "Agents")


class completion(loader):
    pass


completion.init("zyx.lib.client.chat", "completion")


class classify(loader):
    pass


classify.init("zyx.lib.client.functions.classify", "classify")


class code(loader):
    pass


code.init("zyx.lib.client.functions.code", "code")


class create_system_prompt(loader):
    pass


create_system_prompt.init("zyx.lib.client.functions.create_system_prompt", "create_system_prompt")


class extract(loader):
    pass


extract.init("zyx.lib.client.functions.extract", "extract")


class function(loader):
    pass


function.init("zyx.lib.client.functions.function", "function")


class generate(loader):
    pass


generate.init("zyx.lib.client.functions.generate", "generate")


class least_to_most(loader):
    pass


least_to_most.init("zyx.lib.client.functions.least_to_most", "least_to_most")


class optimize_system_prompt(loader):
    pass

optimize_system_prompt.init("zyx.lib.client.functions.optimize_system_prompt", "optimize_system_prompt")


class plan(loader):
    pass


plan.init("zyx.lib.client.functions.plan", "plan")


class self_consistency(loader):
    pass


self_consistency.init("zyx.lib.client.functions.self_consistency", "self_consistency")


class self_refine(loader):
    pass


self_refine.init("zyx.lib.client.functions.self_refine", "self_refine")


class step_back(loader):
    pass


step_back.init("zyx.lib.client.functions.step_back", "step_back")


class tree_of_thought(loader):
    pass


tree_of_thought.init("zyx.lib.client.functions.tree_of_thought", "tree_of_thought")


class query(loader):
    pass


query.init("zyx.lib.client.functions.query", "query")
