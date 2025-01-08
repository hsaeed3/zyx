"""
### zyx

A Python library for `effortlessly` creating and managing complex Generative AI
applications & pipelines, with a `specification on controllable, on-message & autonomous
AI agents`, all through an incredibly non-complex & comfortable API. `zyx` is designed
to be hyper-lightweight, with its only `main` dependencies being `litellm`, `instructor`, & `qdrant-client`.

`zyx` was built with speed, simplicity & developer (my own) experience in mind, and is most easily
usable straight from the top:

```python
import zyx
```

---

hammad saeed | 2025
"""


# [Imports]
__all__ = [
    # logging
    "logging",
    "is_verbose",
    "set_debug",
    "set_verbose",
]


# [Logging]
from . import _logging as logging
from ._logging import set_debug, set_verbose, is_verbose