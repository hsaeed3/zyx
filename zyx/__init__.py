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
    "debug",
    "verbose",
]


# [Logging]
from . import _logging as logging
from ._logging import debug, verbose