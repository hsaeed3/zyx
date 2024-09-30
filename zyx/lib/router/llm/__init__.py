__all__ = [
    "Client",
    "completion",
    "classify",
    "code",
    "extract",
    "function",
    "generate",
    "system_prompt",
]


from .._router import router


from ....client import (
    Client,
    completion,
)


class classify(router):
    pass


classify.init("zyx.resources.completions.base.classify", "classify")


class code(router):
    pass


code.init("zyx.resources.completions.base.code", "code")


class extract(router):
    pass


extract.init("zyx.resources.completions.base.extract", "extract")


class function(router):
    pass


function.init("zyx.resources.completions.base.function", "function")


class generate(router):
    pass


generate.init("zyx.resources.completions.base.generate", "generate")


class system_prompt(router):
    pass


system_prompt.init("zyx.resources.completions.base.system_prompt", "system_prompt")
