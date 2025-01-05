"""
zyx

A Python library for `effortlessly` creating and managing complex Generative AI
applications & pipelines, with a `specification on controllable, on-message & autonomous
AI agents`, all through an incredibly non-complex & comfortable API. `zyx` is designed
to be hyper-lightweight, with its only `main` dependencies being `litellm` & `qdrant-client`.

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
    # [Core Utils & Flags]
    "utils",
    "zyx_verbose",
    "zyx_debug",
]


# ===================================================================
# [Imports]
# ===================================================================

# [Core Utils & Flags]
from . import _utils as utils

# Retrieve Flags, so that if they are set here, they are set globally as well
global zyx_verbose
global zyx_debug
zyx_verbose: bool = utils.zyx_verbose
"""Modules will provide printed console outputs & simple information."""
zyx_debug: bool = utils.zyx_debug
"""Modules will provide extensive & detailed debug information."""
