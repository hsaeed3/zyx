---
title: Parse
icon: lucide/code
---

# `parse()`

??? note "API Reference"

    ??? note "`parse()`"

        ::: zyx.operations.parse.parse
            handler: python

    ??? note "`aparse()`"

        ::: zyx.operations.parse.aparse
            handler: python

---

## Overview

The **parse()** semantic operation is used to extract structured data from a **source** into a **target** type or schema. It treats the input as literal content to parse and returns only the structured output that matches the target.

=== "Sync"

    ```python title="Parsing into a Model"
    from zyx import parse
    from pydantic import BaseModel


    class Invoice(BaseModel):
        id: str
        total: float


    text = "Invoice #A-1045. Total due: $19.99"

    result = parse(
        source=text,
        target=Invoice,
    )


    print(result.output)
    """
    Invoice(id="A-1045", total=19.99)
    """
    ```

=== "Async"

    ```python title="Parsing into a Model"
    from zyx import aparse
    from pydantic import BaseModel


    class Invoice(BaseModel):
        id: str
        total: float


    async def main():
        result = await aparse(
            source="Invoice #A-1045. Total due: $19.99",
            target=Invoice,
        )
    ```

??? tip "Default Target"

    If you do not provide a **`target`**, the operation defaults to `str`, which means it will parse and return the raw primary input as text.

## Streaming

**parse()** can stream outputs as they are generated.

```python title="Streaming a Parsed Output"
from zyx import parse


stream = parse(
    source="Invoice #A-1045. Total due: $19.99",
    target=str,
    stream=True,
)


for chunk in stream.text():
    print(chunk)
```
