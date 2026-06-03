---
title: Validate
icon: lucide/check
---

??? warning "Deprecated API"

    The documentation on this page is no longer maintained in favor of the new ``zyx.beta`` API. You can
    get started by checking out the [beta documentation]{data-preview} instead!

[beta documentation]: .../beta/overview.md

# `validate()`

Parse a **source** into a **target** type and then validate it against one or more **constraints**. You can choose to raise on violations or receive a structured result.

??? note "API Reference"

    ??? note "`validate()`"

        ::: zyx.operations.validate.validate
            handler: python

    ??? note "`avalidate()`"

        ::: zyx.operations.validate.avalidate
            handler: python

---

## Overview

The **validate()** semantic operation parses a **source** into a **target** type and then validates it against one or more **constraints**. You can choose to raise on violations or receive a structured result.

=== "Sync"

    ```python title="Parsing + Constraint Validation"
    from zyx import validate
    from pydantic import BaseModel


    class User(BaseModel):
        name: str
        age: int


    result = validate(
        source="Sam is 15 years old.",
        target=User,
        constraints=[
            "age must be at least 18",
        ],
        raise_on_error=False,
    )


    print(result.output)
    print(result.violations.violations)
    """
    User(name="Sam", age=15)
    [ConstraintViolation(constraint="age must be at least 18", reason="age is 15")]
    """
    ```

=== "Async"

    ```python title="Parsing + Constraint Validation"
    from zyx import avalidate
    from pydantic import BaseModel


    class User(BaseModel):
        name: str
        age: int


    async def main():
        result = await avalidate(
            source="Sam is 15 years old.",
            target=User,
            constraints=[
                "age must be at least 18",
            ],
            raise_on_error=False,
        )
    ```

??? tip "Raise on Error"

    Set **`raise_on_error=True`** to raise an `AssertionError` when any constraint is violated. Set it to False to receive a `ValidationResult` with violations instead.
