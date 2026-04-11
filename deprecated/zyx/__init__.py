"""zyx

> A fun **"anti-framework"** for doing useful things with agents and LLMs.

## Quick Start

### Generating Content

```python
from zyx import make

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

result = make(
    target=str,
    context="What is the weather in Tokyo?",
    tools=[get_weather],
)
```

### Streaming

```python
from zyx import make

stream = make(context="Write a haiku about the weather.", stream=True)

for chunk in stream.text():
    print(chunk, end="", flush=True)
```

---

### Editing Values

```python
from zyx import edit

data = {
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
}

result = edit(
    target=data,
    context="Change the name to 'Jane Doe'.",
    merge=False,
)
```
"""

# --- core components & objects ---
from .attachments import Attachment, paste, attach
from .context import Context, create_context
from .targets import Target, target
from .result import Result
from .stream import Stream

# --- tools / prebuilt toolsets ---
from .tools import Tool, tool, Code, Memory, memories

# --- semantic operations ---
from .operations.edit import aedit, edit
from .operations.expressions import expr
from .operations.make import amake, make
from .operations.parse import aparse, parse
from .operations.query import aquery, query
from .operations.run import arun, run
from .operations.select import aselect, select
from .operations.validate import avalidate, validate

__all__ = (
    # --- core components & objects ---
    "Attachment",
    "paste",
    "attach",
    "Context",
    "create_context",
    "Target",
    "target",
    "Result",
    "Stream",
    # --- tools / prebuilt toolsets ---
    "Tool",
    "tool",
    "Code",
    "Memory",
    "memories",
    # --- semantic operations ---
    "aedit",
    "edit",
    "expr",
    "amake",
    "make",
    "aparse",
    "parse",
    "aquery",
    "query",
    "arun",
    "run",
    "aselect",
    "select",
    "avalidate",
    "validate",
)
