"""zyx.core.stores.utils"""

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
        The value to convert to a stable string representation.

    Returns
    -------
    str
        A stable string representation of the value.

    Examples
    --------
    ```python
        >>> stable_string("hello")
        'hello'
        >>> stable_string({"b": 2, "a": 1})
        '{"a": 1, "b": 2}'
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Person:
        ...     name: str
        ...     age: int
        >>> stable_string(Person("Alice", 30))
        '{"age": 30, "name": "Alice"}'
    ```
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if is_dataclass(value):
        return json.dumps(
            asdict(value), ensure_ascii=False, sort_keys=True
        )
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

    This function handles common Python objects and converts them to
    JSON-compatible formats. It handles primitives, dataclasses, and
    Pydantic models.

    Parameters
    ----------
    value : Any
        The value to convert to a JSON-serializable format.

    Returns
    -------
    Any
        A JSON-serializable representation of the value.

    Examples
    --------
    ```python
        >>> json_ready("hello")
        'hello'
        >>> json_ready(42)
        42
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Person:
        ...     name: str
        ...     age: int
        >>> json_ready(Person("Alice", 30))
        {'name': 'Alice', 'age': 30}
    ```
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


def generate_content_key(
    content: Any, metadata: Dict[str, Any] | None = None
) -> str:
    """Generate a stable hash key for content and optional metadata.

    This function creates a deterministic hash from content and metadata,
    useful for deduplication and content-based addressing.

    Parameters
    ----------
    content : Any
        The content to generate a key for.
    metadata : Dict[str, Any] | None
        Optional metadata to include in the key generation.

    Returns
    -------
    str
        A 16-character hexadecimal hash string.

    Examples
    --------
    ```python
        >>> generate_content_key("hello world")
        'b94d27b9934d3e08'
        >>> generate_content_key("hello", {"source": "user"})
        '3afc79b6d3c05b9c'
    ```
    """
    base = stable_string(content)
    meta = stable_string(metadata or {})
    digest = hashlib.sha256(f"{base}:{meta}".encode("utf-8")).hexdigest()
    return digest[:16]
