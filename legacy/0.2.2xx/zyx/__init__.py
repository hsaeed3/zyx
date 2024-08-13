__all__ = [
    "BaseModel",
    "Field",
    "logger",
    "db",
    "code",
    "extract",
    "generate",
    "classify",
    "completion",
    "function",
    "toolkits",
    "zyxModuleLoader",
]

import builtins
from rich import print

builtins.print = print
from .core.ext import zyxModuleLoader, BaseModel, Field
from .core import toolkits as toolkits


class logger(zyxModuleLoader):
    pass


logger.init("loguru", "logger")


class db(zyxModuleLoader):
    pass


db.init("zyx.core.data", "database")


class completion(zyxModuleLoader):
    pass


completion.init("zyx.core.client", "completion")


class code(zyxModuleLoader):
    pass


code.init("zyx.core.client", "code")


class extract(zyxModuleLoader):
    pass


extract.init("zyx.core.client", "extract")


class generate(zyxModuleLoader):
    pass


generate.init("zyx.core.client", "generate")


class classify(zyxModuleLoader):
    pass


classify.init("zyx.core.client", "classify")


class function(zyxModuleLoader):
    pass


function.init("zyx.core.client", "function")
