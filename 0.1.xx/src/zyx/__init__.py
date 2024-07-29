# zyx ==============================================================================

__all__ = [
    "batch_completion",
    "completion",
    "embedding",
    "Inference",
    "ollama_embedding",
    "db",
    "Memory",
    "Qdrant",
    "logger",
    "console",
    "BaseModel",
    "Field",
    "tqdm",
    "tqdm_notebook",
]

from .ext.ai import batch_completion as batch_completion
from .ext.ai import completion as completion
from .ext.ai import embedding as embedding
from .ext.ai import Inference as Inference
from .ext.ai import ollama_embedding as ollama_embedding
from .ext.data import db as db
from .ext.data import memory as Memory
from .ext.data import qdrant as Qdrant
from .ext.util import logger as logger
from .ext.util import console as console
from .ext.util import BaseModel as BaseModel
from .ext.util import Field as Field
from .ext.util import tqdm as tqdm
from .ext.util import tqdm_notebook as tqdm_notebook

from rich import print
import builtins as __builtins__

__builtins__.print = print
