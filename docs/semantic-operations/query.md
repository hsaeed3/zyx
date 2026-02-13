---
title: Query
icon: lucide/book
---

# `query()`

??? note "API Reference"

    ??? note "`query()`"

        ::: zyx.operations.query.query
            handler: python

    ??? note "`aquery()`"

        ::: zyx.operations.query.aquery
            handler: python

---

## Overview

The **query()** semantic operation answers questions using a grounded **source**. It only uses the provided source content and returns a structured output that matches the **target** type or schema.

=== "Sync"

    ```python title="Grounded Query"
    from zyx import query
    from pydantic import BaseModel


    class PolicyAnswer(BaseModel):
        max_refunds: int | None
        notes: str | None


    policy_text = "Refunds are allowed up to 30 days after purchase."

    result = query(
        source=policy_text,
        target=PolicyAnswer,
        context="How many days do customers have to request a refund?",
    )


    print(result.output)
    """
    PolicyAnswer(max_refunds=30, notes=None)
    """
    ```

=== "Async"

    ```python title="Grounded Query"
    from zyx import aquery
    from pydantic import BaseModel


    class PolicyAnswer(BaseModel):
        max_refunds: int | None
        notes: str | None


    async def main():
        result = await aquery(
            source="Refunds are allowed up to 30 days after purchase.",
            target=PolicyAnswer,
            context="How many days do customers have to request a refund?",
        )
    ```

??? tip "Unknowns"

    If the answer is not supported by the **source**, query will return `None` for the relevant fields when the schema allows it.

## Streaming

**query()** can stream outputs as they are generated.

```python title="Streaming a Query"
from zyx import query


stream = query(
    source="Refunds are allowed up to 30 days after purchase.",
    target=str,
    context="What does the policy say about refunds?",
    stream=True,
)


for chunk in stream.text():
    print(chunk)
```
