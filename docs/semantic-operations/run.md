---
title: Run
icon: lucide/rocket
---

# `run()`

Run an agent or model on an **arbitrary task**, and optionally return a **target** value, or a representation of the completion of the task.

??? note "API Reference"

    ??? note "`run()`"

        ::: zyx.operations.run.run
            handler: python

    ??? note "`arun()`"

        ::: zyx.operations.run.arun
            handler: python

---

## Overview

The **run()** semantic operation executes a **task** and requires the model to signal completion via a tool call. Unlike other operations, **run()** is goal-centric: completion is not inferred from plain text. If you provide a **target**, the completion tool must include a result that validates against that schema.

If **task** is `None`, `run()` falls back to **make()** behavior and returns a standard `Result[T]` for the target.

=== "Sync"

    ```python title="Basic Task"
    from zyx import run


    def get_weather(city: str) -> str:
        return f"{city}: 72F and sunny"


    result = run(
        task="Check the weather in Los Angeles and summarize it.",
        tools=[get_weather],
    )


    print(result.completion.summary)
    """
    Checked the weather in Los Angeles and summarized the result.
    """
    ```

=== "Async"

    ```python title="Basic Task"
    from zyx import arun


    def get_weather(city: str) -> str:
        return f"{city}: 72F and sunny"


    async def main():
        result = await arun(
            task="Check the weather in Los Angeles and summarize it.",
            tools=[get_weather],
        )
    ```

??? tip "Structured Results"

    If you provide a **target**, the model must supply a matching `result` when completing the task.

    ```python
    from zyx import run
    from pydantic import BaseModel


    class Todo(BaseModel):
        item: str
        priority: int


    result = run(
        task="Create a single todo for today's top priority.",
        target=Todo,
    )

    print(result.output)
    """
    Todo(item="Finish quarterly report", priority=1)
    """
    ```

## Verification

You can add deterministic checks for the completion payload. A **verifier** runs once and can fail the task; **final_answer_checks** run after verification and can apply multiple checks.

```python title="Verification"
from zyx import run


def verifier(completion, ctx):
    if completion.status != "success":
        return "Task did not complete successfully."
    if completion.result is None:
        return "Missing result."
    return True


result = run(
    task="Summarize the document and produce a JSON summary.",
    target=dict,
    verifier=verifier,
)
```

## Planning Mode

Set **plan=True** to have the model generate and execute a deterministic plan. You can bound planning with **planning_interval** (maximum tool calls per plan) and **max_steps** (total tool calls across the run).

```python title="Plan-Based Run"
from zyx import run


def fetch(url: str) -> str:
    return f"Downloaded: {url}"


result = run(
    task="Fetch the release notes and summarize the key changes.",
    tools=[fetch],
    plan=True,
    planning_interval=4,
    max_steps=10,
)
```

??? tip "Completion Behavior"

    By default, **require_completion=True** and the model must call the completion tool. If you set **require_completion=False**, `run()` will return even if the task did not complete; in that case, `result.completion.status` will be `"skipped"`.

## Streaming

**run()** supports streaming **only** when a **target** is provided and **plan=False**.

```python title="Streaming a Run"
from zyx import run


stream = run(
    task="Draft a short checklist for deploying a web app.",
    target=list[str],
    stream=True,
)


for chunk in stream.partial():
    print(chunk)


final = stream.finish()
print(final.output)
```
