# zyx ===============================================

from ._loader import UtilLazyLoader


class BaseModel(UtilLazyLoader):
    pass


BaseModel.init("pydantic.main", "BaseModel")


class Field(UtilLazyLoader):
    pass


Field.init("pydantic.fields", "Field")


class logger(UtilLazyLoader):
    pass


logger.init("loguru", "logger")


class console(UtilLazyLoader):
    pass


console.init("rich.console", "Console")


class tqdm(UtilLazyLoader):
    pass


tqdm.init("tqdm", "tqdm")


class tqdm_notebook(UtilLazyLoader):
    pass


tqdm_notebook.init("tqdm.notebook", "tqdm_notebook")


class lightning(UtilLazyLoader):
    pass


lightning.init("zyx.functions.lightning", "lightning")

# ====================================================
