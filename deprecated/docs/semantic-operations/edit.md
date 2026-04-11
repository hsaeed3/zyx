---
title: Edit
icon: lucide/pencil-line
---

# `edit()`

Edit a `target` value using natural language instructions, along with **type-specific**, **field/content selective**, and **planning** capabilities.

??? note "API Reference"

    ??? note "`edit()`"

        ::: zyx.operations.edit.edit
            handler: python

    ??? note "`aedit()`"

        ::: zyx.operations.edit.aedit
            handler: python

---

## Overview

The **edit()** semantic operation is one of the most powerful and in-depth operations provided by `ZYX`. It allows you to pass some **target** data and have the model perform a single or series of edits on it, with the optional ability to plan  the edits first as well as merge into the original data.

The biggest upsides of the **edit()** operation include 2 key points:

1. The ability to perform *selective* edits, where the model only generates the parts of the data that need to be edited. (Currently only supported for string and mapping-like (dict/basemodel/dataclass/etc) data types). Which allows the model to make more granular and cost-efficient edits.

2. The ability to *plan* edits and generate them *iteratively* (one edit at a time), allowing for greater control and accuracy over the edits.

??? tip "Edit Strategies"

    When it is mentioned that a model can perform selective edits based on compatible targets; this is a reference to the various *edit strategies* that `ZYX` provides.

    For example, in the case of strings, a model can generate string diffs or replacements based on specific anchors or selections within the text.

=== "Sync"

    ```python title="Basic Edit"
    from zyx import edit


    data = {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com",
    } # (1)!


    result = edit(
        target=data,
        context="Change the name to 'Jane Doe'.",
        merge=False, # (2)!
    )


    print(result.output)
    """
    {
        "name": "Jane Doe",
        "age": None,
        "email": None
    }
    """
    ```

    1. The **`target`** parameter when using edit can be a value of any standard target/data type.

    2. The **`merge`** parameter allows to merge the edits into the original data, or just return the new content.

=== "Async"

    ```python title="Basic Edit"
    from zyx import aedit


    data = {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com",
    }


    async def main():
        result = await aedit(
            target=data,
            context="Change the name to 'Jane Doe'.",
            merge=False,
        )
    ```

## Selective Edits

By default, the **edit()** operation performs a **selective** edit (if a supported target is provided). Which means the model only generates the parts of the data that need to be edited.

We can best understand this by example:

=== "Selective"

    ```python title="Selective Edit"
    from zyx import edit
    from pydantic import BaseModel


    class Product(BaseModel):
        name : str = "Apple"
        description : str = "A delicious fruit that is red and sweet."
        price : float = 1.99
        quantity : int = 10


    result = edit(
        target=Product(),
        context="We've got 15 of them right now.",
        selective=True,
        merge=False
    )


    print(result.output)
    """
    Product(name=None, description=None, price=None, quantity=15)
    """
    ```

=== "Non-Selective"

    ```python title="Non-Selective Edit"
    from zyx import edit
    from pydantic import BaseModel


    class Product(BaseModel):
        name : str = "Apple"
        description : str = "A delicious fruit that is red and sweet."
        price : float = 1.99
        quantity : int = 10


    result = edit(
        target=Product(),
        context="We've got 15 of them right now.",
        selective=False, # (1)!
        merge=False
    )


    print(result.output) # (2)!
    """
    Product(name="Apple", description="A delicious fruit that is red and sweet.", price=1.99, quantity=15)
    """
    ```

    1. We set *`selective`* to False.

    2. If its not immediately clear, as we have set *`merge`* to False, the model should only return the new content, and in this case all fields were returned, meaning the model performed a full re-generation of the data.
