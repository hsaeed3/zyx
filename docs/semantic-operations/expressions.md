---
title: Expressions
icon: lucide/list-check
---

# `expr()`

??? note "API Reference"

    ??? note "`expr()`"

        ::: zyx.operations.expressions.expr
            handler: python

## Overview

The **expr()** semantic operation is a simple operation that allows you to use Pythonic dunder methods or expressions to perform evaluations against a given **source**.

Unlike all other semantic operations, the **expr()** operation cannot be asynchronously and has a locked **`target`** type, based on the expression being used.

```python title="Using expr()"
from zyx import expr

if "hate speech" in expr("you suck"):
    print("The statement contains hate speech.")
else:
    print("The statement does not contain hate speech.")
"""
The statement contains hate speech.
"""
```

## Supported Expressions

Flip through the following table to get a better idea of the various expressions that can be used with **`expr()`**.

=== "Boolean Expressions"

    ```python title="=="
    from zyx import expr

    if expr("the weather is sunny today") == "positive":
        print("positive")
    """
    positive
    """
    ```

=== "Inclusion Expressions"

    ```python title="in"
    from zyx import expr

    if "hate speech" in expr("you suck"):
        print("The statement contains hate speech.")
    """
    The statement contains hate speech.
    """
    ```

=== "Greater Than / Less Than"

    ```python title="contains"
    from zyx import expr

    if expr("he has 43 active issues") > 30:
        print("He has more than 30 active issues.")
    """
    He has more than 30 active issues.
    """
    ```
