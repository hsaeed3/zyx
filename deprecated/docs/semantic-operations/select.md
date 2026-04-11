---
title: Run
icon: lucide/goal
---

# `select()`

Select **one or more options** from a given list or type representing a set of options.

??? note "API Reference"

    ??? note "`select()`"

        ::: zyx.operations.select.select
            handler: python

    ??? note "`aselect()`"

        ::: zyx.operations.select.aselect
            handler: python

---

## Overview

The **select()** semantic operation is used to select one more options from a given list or type representing a set of options that a model can choose from.

=== "Sync"

    ```python title="Single Classification"
    from zyx import select


    result = select(
        target=["red", "green", "blue"], # (1)!
        context="What is the color of the sky?",
    )


    print(result.output)
    """
    "blue"
    """
    ```

    1. The **`target`** parameter when using select, can be a list of any compatible types (str, int, BaseModel, etc.), or a Literal[...], Enum, or Union[..., ...] type.

=== "Async"

    ```python title="Single Classification"
    from zyx import aselect


    async def main():
        result = await aselect(
            target=["red", "green", "blue"],
            context="What is the color of the sky?",
        )
    ```

## Multi-Label Selection

When running this operation, you can choose between setting the **`multi_label`** parameter to True or False, which allows for single or multiple selection respectively.

```python title="Multi-Label Selection"
from zyx import select


result = select(
    target=["red", "black", "blue", "yellow", "purple"],
    context="What colors make green?",
    multi_label=True,
)


print(result.output)
"""
["blue", "yellow"]
"""
```

## Reasoning

Optionally, you can set the **`include_reason`** parameter to True, to allow the model to provide a textual reason for it's selection.

```python title="Including Reasoning"
from zyx import select


result = select(
    ["positive", "negative"],
    context="This produce is okay.",
    include_reason=True,
)


print(result.output.selection)
print(result.output.reason)
"""
positive
The statement indicates that the produce is acceptable or satisfactory.
"""
```
