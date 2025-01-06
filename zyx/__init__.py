"""
zyx

A Python library for `effortlessly` creating and managing complex Generative AI
applications & pipelines, with a `specification on controllable, on-message & autonomous
AI agents`, all through an incredibly non-complex & comfortable API. `zyx` is designed
to be hyper-lightweight, with its only `main` dependencies being `litellm`, `instructor`, & `qdrant-client`.

`zyx` was built with speed, simplicity & developer (my own) experience in mind, and is most easily
usable straight from the top:

---

*Chat Completions w/ Easy Structured Outputs & Automatic Tool Calling*

```python
import zyx

response = zyx.create.completion("hi")
```

---

hammad saeed | 2025
"""


__all__ = [
    # [Core & Utility | Flags]
    # Logging
    "logging",
    # Flags
    "zyx_verbose",
    "zyx_debug",
]


# [Imports]
# [Configure Logging & Flags]
from . import _logging as logging
global zyx_verbose
global zyx_debug
zyx_verbose: bool = logging.zyx_verbose
zyx_debug: bool = logging.zyx_debug
