"""zyx.core.memory.utils"""

from functools import lru_cache
import json
import hashlib
from typing import List
import struct

__all__ = [
    "serialize_float32_vector_to_bytes",
    "deserialize_bytes_to_float32_vector",
    "generate_content_id",
]


@lru_cache(maxsize=None)
def _generate_content_id(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def generate_content_id(content: str, metadata: dict | None = None) -> str:
    """Generate unique ID for content"""
    data = f"{content}:{json.dumps(metadata or {}, sort_keys=True)}"
    return _generate_content_id(data)


@lru_cache(maxsize=None)
def _serialize_float32_vector_to_bytes(vector_tuple: tuple) -> bytes:
    return struct.pack(f"{len(vector_tuple)}f", *vector_tuple)


def serialize_float32_vector_to_bytes(vector: List[float]) -> bytes:
    """Serialize a list of float32 numbers to bytes."""
    return _serialize_float32_vector_to_bytes(tuple(vector))


@lru_cache(maxsize=None)
def deserialize_bytes_to_float32_vector(data: bytes) -> List[float]:
    """Deserialize bytes back to a list of float32 numbers."""
    num_floats = len(data) // 4  # Each float32 is 4 bytes
    return list(struct.unpack(f"{num_floats}f", data))
