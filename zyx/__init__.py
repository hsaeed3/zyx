__all__ = [
    # core
    "BaseModel",
    "Field",
    "logger",
    "zyxModuleLoader",
    # model application
    "app",
    # client -- core
    "completion",
    "embedding",
    "image",
    "speak",
    "transcribe",
    # client -- chat
    "chat",
    # client -- functions
    "chainofthought",
    "classify",
    "code",
    "extract",
    "function",
    "generate",
    # client -- data
    "Rag",
    # agents
    "Agents",
    "tools",
]

# -- core --
from .core.main import BaseModel, Field, zyxModuleLoader


class logger(zyxModuleLoader):
    pass


logger.init("loguru", "logger")


# -- model application --
class app(zyxModuleLoader):
    pass


app.init("zyx.client.chat.main", "app")


# -- client -- core --
class completion(zyxModuleLoader):
    pass


completion.init("zyx.client.main", "completion")


class embedding(zyxModuleLoader):
    pass


embedding.init("litellm.main", "embedding")


class image(zyxModuleLoader):
    pass


image.init("zyx.client.multimodal", "image")


class speak(zyxModuleLoader):
    pass


speak.init("zyx.client.multimodal", "speak")


class transcribe(zyxModuleLoader):
    pass


transcribe.init("zyx.client.multimodal", "transcribe")


# -- client -- chat --
class chat(zyxModuleLoader):
    pass


chat.init("zyx.client.chat.chat", "chat")


# -- client -- functions --
class chainofthought(zyxModuleLoader):
    pass


chainofthought.init("zyx.client.functions.chainofthought", "chainofthought")


class classify(zyxModuleLoader):
    pass


classify.init("zyx.client.functions.classify", "classify")


class code(zyxModuleLoader):
    pass


code.init("zyx.client.functions.code", "code")


class extract(zyxModuleLoader):
    pass


extract.init("zyx.client.functions.extract", "extract")


class function(zyxModuleLoader):
    pass


function.init("zyx.client.functions.function", "function")


class generate(zyxModuleLoader):
    pass


generate.init("zyx.client.functions.generate", "generate")


# -- client -- data --
class Rag(zyxModuleLoader):
    pass


Rag.init("zyx.data.rag", "Rag")


# -- agents --
class Agents(zyxModuleLoader):
    pass


Agents.init("zyx.agents.main", "Agents")


from .agents import tools as tools
