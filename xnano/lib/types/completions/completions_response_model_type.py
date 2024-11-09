# completion response model

from pydantic import BaseModel
from typing import Any


# model
class CompletionsResponseModelType(BaseModel):
    """Completion response model."""

    response: Any

