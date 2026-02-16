---
title: Make
icon: lucide/sparkle
---

# `make()`

Generate text and **structured outputs** based on a given `target`, with the ability to create **synthetic content**.

??? note "API Reference"

    ??? note "`make()`"

        ::: zyx.operations.make.make
            handler: python

    ??? note "`amake()`"

        ::: zyx.operations.make.amake
            handler: python

---

## Overview

The **make()** semantic operation is used to generate content and structured outputs based on a given **target** type or value.

=== "Sync"

    ```python title="Generating a Structured Output"
    from zyx import make
    from pydantic import BaseModel


    class User(BaseModel):
        name : str
        age : int


    result = make(
        target=User,
        context="John is 25 years old."
    )


    print(result.output)
    """
    User(name="John", age=25)
    """
    ```

=== "Async"

    ```python title="Generating a Structured Output"
    from zyx import amake
    from pydantic import BaseModel


    class User(BaseModel):
        name : str
        age : int


    async def main():
        result = await amake(
            target=User,
            context="John is 25 years old."
        )
    ```

??? tip "Did you know?"

    **make()** can be called with or without **context** allowing for model determined synthetic content generation.

    ```python
    result = make(User) # (1)!

    print(result.output)
    """
    name='Leona Frost' age=28
    """
    ```

    1. In this case, only the **`target`** was provided.


## Streaming

**make()** also supports streaming the output of the operation as it is generated.

```python title="Streaming the Output"
from zyx import make


stream = make(
    target=list[str],
    context="Write a haiku about the weather.",
    stream=True # (1)!
)


for chunk in stream.partial(): # (2)!
    print(chunk)
"""
['Whispers of the breeze,', 'Clouds']
['Whispers of the breeze,', 'Clouds dance in the silver sky,']
['Whispers of the breeze,', 'Clouds dance in the silver sky,', 'Raindrops kiss the earth.']
['Whispers of the breeze,', 'Clouds dance in the silver sky,', 'Raindrops kiss the earth.']
['Whispers of the breeze,', 'Clouds dance in the silver sky,', 'Raindrops kiss the earth.']
"""
```

1. We set the **`stream`** parameter to True to enable streaming.

2. A stream can be iterated over using the **`text()`** or **`partial()`** methods. In this case, we are using a non-string structured output, so we use **`partial()`**.
