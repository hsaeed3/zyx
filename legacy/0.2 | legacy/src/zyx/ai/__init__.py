# zyx ==============================================================================

__all__ = [
    "cast",
    "classify",
    "completion",
    "CompletionClient",
    "embeddings",
    "extract",
    "function",
    "instructor_completion",
    "generate",
    "inference",
]

from ..core import _UtilLazyLoader

class completion(_UtilLazyLoader):
    pass
completion.init("zyx.ai._completion", "completion")

from ._completion import CompletionClient as CompletionClient

class embeddings(_UtilLazyLoader):
    pass
embeddings.init("zyx.ai._embeddings", "embeddings")

class cast(_UtilLazyLoader):
    pass
cast.init("marvin.ai.text", "cast")

class extract(_UtilLazyLoader):
    pass
extract.init("marvin.ai.text", "extract")

class classify(_UtilLazyLoader):
    pass
classify.init("marvin.ai.text", "classify")

class function(_UtilLazyLoader):
    pass
function.init("marvin.ai.text", "fn")

class generate(_UtilLazyLoader):
    pass
generate.init("marvin.ai.text", "generate")

class inference(_UtilLazyLoader):
    pass
inference.init("huggingface_hub.inference._client", "InferenceClient")

class instructor_completion(_UtilLazyLoader):
    pass
instructor_completion.init("zyx.ai._instructor_completion", "instructor_completion")