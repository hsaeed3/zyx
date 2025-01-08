"""
zyx.types.multimodal

Types used for multimodal generation and data.
"""

from __future__ import annotations

# [Imports]
from typing import Union
from pydantic import BaseModel, model_serializer
from pathlib import Path
from base64 import b64encode, b64decode


# [Image Type]
class Image(BaseModel):
    value: Union[str, bytes, Path]

    @model_serializer
    def serialize_model(self):
        if isinstance(self.value, (Path, bytes)):
            return b64encode(self.value.read_bytes() if isinstance(self.value, Path) else self.value).decode()

        if isinstance(self.value, str):
            try:
                if Path(self.value).exists():
                    return b64encode(Path(self.value).read_bytes()).decode()
            except Exception:
                # Long base64 string can't be wrapped in Path, so try to treat as base64 string
                pass

            # String might be a file path, but might not exist
            if self.value.split(".")[-1] in ("png", "jpg", "jpeg", "webp"):
                raise ValueError(f"File {self.value} does not exist")

            try:
                # Try to decode to check if it's already base64
                b64decode(self.value)
                return self.value
            except Exception:
                raise ValueError("Invalid image data, expected base64 string or path to image file") from Exception
