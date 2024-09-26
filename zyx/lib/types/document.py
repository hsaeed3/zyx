from pydantic import BaseModel
from typing import Any, Dict


class Document(BaseModel):
    content: Any
    metadata: Dict[str, Any]