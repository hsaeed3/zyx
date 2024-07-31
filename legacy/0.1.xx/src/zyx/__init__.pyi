# zyx ==============================================================================

from litellm.main import (
    batch_completion_models_all_responses as batch_completion_models_all_responses_type,
)
from .functions.completion import completion as completion_type
from .functions.lightning import lightning as lightning_type
from litellm.main import embedding as embedding_type
from huggingface_hub.inference._client import InferenceClient as InferenceClient_type
from ollama import embeddings as ollama_embedding_type
import sqlite3 as sqlite3_type
from mem0.memory.main import Memory as Memory_type
from qdrant_client.qdrant_client import QdrantClient as QdrantClient_type
from loguru import logger as logger_type
from pydantic.main import BaseModel as BaseModel_type
from pydantic.fields import Field as Field_type
from rich.console import Console as console_type
from tqdm import tqdm as tqdm_type
from tqdm.notebook import tqdm_notebook as tqdm_notebook_type

# ---------------------------------------------------

batch_completion = batch_completion_models_all_responses_type
completion = completion_type
lightning = lightning_type

# ---------------------------------------------------

embedding = embedding_type
Inference = InferenceClient_type
ollama_embedding = ollama_embedding_type

# ---------------------------------------------------

db = sqlite3_type
memory = Memory_type
qdrant = QdrantClient_type

# ---------------------------------------------------

logger = logger_type
console = console_type
BaseModel = BaseModel_type
Field = Field_type
tqdm = tqdm_type
tqdm_notebook = tqdm_notebook_type

# ---------------------------------------------------
