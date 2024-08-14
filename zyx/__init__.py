__all__ = [
    "BaseModel",
    "Field",
    "logger",
    "cli",
    "chainofthought",
    "classify",
    "completion",
    "code",
    "extract",
    "function",
    "generate",
    "zyxModuleLoader",
]

# --- zyx ----------------------------------------------------------------

from .core.ext import BaseModel, Field, zyxModuleLoader


class cli(zyxModuleLoader):
    pass


cli.init("zyx.client.app", "cli")


class chainofthought(zyxModuleLoader):
    pass


chainofthought.init("zyx.client.fn", "chainofthought")


class classify(zyxModuleLoader):
    pass


classify.init("zyx.client.fn", "classify")


class code(zyxModuleLoader):
    pass


code.init("zyx.client.fn", "code")


class extract(zyxModuleLoader):
    pass


extract.init("zyx.client.fn", "extract")


class function(zyxModuleLoader):
    pass


function.init("zyx.client.fn", "function")


class generate(zyxModuleLoader):
    pass


generate.init("zyx.client.fn", "generate")


class completion(zyxModuleLoader):
    pass


completion.init("zyx.client.main", "completion")


class logger(zyxModuleLoader):
    pass


logger.init("loguru", "logger")
