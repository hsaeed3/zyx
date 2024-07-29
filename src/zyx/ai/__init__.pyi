# zyx ==============================================================================

__all__ = [
    "cast",
    "classify",
    "completion",
    "CompletionClient",
    "embeddings",
    "extract",
    "function",
    "generate",
    "inference",
    "instructor_completion",
]

from ._completion import completion as completion
from ._completion import CompletionClient as CompletionClient
from ._embeddings import embeddings as embeddings
from ._instructor_completion import instructor_completion as instructor_completion

from huggingface_hub.inference._client import InferenceClient as inference

from marvin.ai.text import cast as cast
from marvin.ai.text import extract as extract
from marvin.ai.text import classify as classify
from marvin.ai.text import fn as function
from marvin.ai.text import generate as generate