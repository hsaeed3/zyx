# zyx.types.completion_response_model
# completion response model

from pydantic import BaseModel
from typing import Any


# model
class CompletionResponseModel(BaseModel):
    """Completion response model."""

    response: Any

