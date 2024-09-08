__all__ = [
    "Agents",
    "completion",
    "embedding",
    "image",
    "llm",
    "logger",
    "Memory",
    "speak",
    "tools",
    "terminal",
    "transcribe",
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
]

from ._client.utils.loader import Loader
from ._client import tools
from ._client.llm import (
    classify,
    code,
    completion,
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
)


class Agents(Loader):
    pass


Agents.init("zyx._client.agents", "Agents")


class completion(Loader):
    pass


completion.init("zyx._client.completion", "completion")


class embedding(Loader):
    pass


embedding.init("litellm.main", "embedding")


class image(Loader):
    pass


image.init("zyx._client.multimodal", "image")



class logger(Loader):
    pass


logger.init("loguru", "logger")


class Memory(Loader):
    pass


Memory.init("zyx._client.memory", "Memory")


class speak(Loader):
    pass


speak.init("zyx._client.multimodal", "speak")


class terminal(Loader):
    pass


terminal.init("zyx._client.terminal", "terminal")


class transcribe(Loader):
    pass


transcribe.init("zyx._client.multimodal", "transcribe")
