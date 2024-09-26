__all__ = [
    "Agents",
    "Client",
    "completion",

    "classify",
    "code",
    "create_system_prompt",
    "extract",
    "function",
    "generate",
    "Judge",
    "least_to_most",
    "optimize_system_prompt",
    "plan",
    "query",
    "self_consistency",
    "self_refine",
    "solve",
    "step_back",
    "tree_of_thought",
    "verifier",

    "terminal",
]


from .._loader import loader


class Agents(loader):
    pass


Agents.init("zyx.lib.completions.agents", "Agents")


class Client(loader):
    pass


Client.init("zyx.lib.completions.client", "Client")


class completion(loader):
    pass


completion.init("zyx.lib.completions.client", "completion")


class classify(loader):
    pass


classify.init("zyx.lib.completions.resources.classify", "classify")


class code(loader):
    pass


code.init("zyx.lib.completions.resources.code", "code")


class create_system_prompt(loader):
    pass


create_system_prompt.init("zyx.lib.completions.resources.create_system_prompt", "create_system_prompt")


class extract(loader):
    pass


extract.init("zyx.lib.completions.resources.extract", "extract")


class function(loader):
    pass


function.init("zyx.lib.completions.resources.function", "function")


class generate(loader):
    pass


generate.init("zyx.lib.completions.resources.generate", "generate")


class Judge(loader):
    pass


Judge.init("zyx.lib.completions.resources.judge", "Judge")


class least_to_most(loader):
    pass


least_to_most.init("zyx.lib.completions.resources.least_to_most", "least_to_most")


class optimize_system_prompt(loader):
    pass


optimize_system_prompt.init("zyx.lib.completions.resources.optimize_system_prompt", "optimize_system_prompt")


class plan(loader):
    pass


plan.init("zyx.lib.completions.resources.plan", "plan")


class query(loader):
    pass


query.init("zyx.lib.completions.resources.query", "query")


class self_consistency(loader):
    pass


self_consistency.init("zyx.lib.completions.resources.self_consistency", "self_consistency")


class self_refine(loader):
    pass


self_refine.init("zyx.lib.completions.resources.self_refine", "self_refine")


class solve(loader):
    pass


solve.init("zyx.lib.completions.resources.solve", "solve")


class step_back(loader):
    pass


step_back.init("zyx.lib.completions.resources.step_back", "step_back")


class tree_of_thought(loader):
    pass


tree_of_thought.init("zyx.lib.completions.resources.tree_of_thought", "tree_of_thought")


class terminal(loader):
    pass


terminal.init("zyx.lib.app", "terminal")


class verifier(loader):
    pass


verifier.init("zyx.lib.completions.resources.judge", "verifier")

