"""zyx.core.memory.utils"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

__all__ = ["generate_content_key", "stable_string", "json_ready"]


def stable_string(value: Any) -> str:
    """Convert any value to a stable string representation for hashing/embedding.

    Parameters
    ----------
    value : Any
        The value to convert.

    Returns
    -------
    str
        A stable string representation of the value.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if is_dataclass(value):
        return json.dumps(asdict(value), ensure_ascii=False, sort_keys=True)
    if hasattr(value, "model_dump"):
        try:
            return json.dumps(
                value.model_dump(),
                ensure_ascii=False,
                sort_keys=True,  # type: ignore[attr-defined]
            )
        except Exception:  # pragma: no cover
            pass
    return str(value)


def json_ready(value: Any) -> Any:
    """Convert a value to a JSON-serializable representation.

    Parameters
    ----------
    value : Any
        The value to convert.

    Returns
    -------
    Any
        A JSON-serializable representation of the value.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (dict, list)):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass
    return value


def generate_content_key(content: Any, metadata: Dict[str, Any] | None) -> str:
    """Generate a stable hash key for content and metadata.

    Parameters
    ----------
    content : Any
        The content to generate a key for.
    metadata : Dict[str, Any] | None
        Optional metadata to include in the key.

    Returns
    -------
    str
        A 16-character hex hash of the content and metadata.
    """
    base = stable_string(content)
    meta = stable_string(metadata or {})
    digest = hashlib.sha256(f"{base}:{meta}".encode("utf-8")).hexdigest()
    return digest[:16]
