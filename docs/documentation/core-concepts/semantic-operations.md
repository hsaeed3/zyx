---
title: Semantic Operations
icon: lucide/sparkles
---

# Semantic Operations

## What are Semantic Operations?

The core generative, or LLM-powered methods within `ZYX` are called **semantic operations**. A semantic operation is a function that leverages language models to:

1. Take in some input **[Context](context.md){ data-preview }** (prompts, instructions, message history, data) and a **[Source](targets.md#sources){ data-preview }** for specific operations.
2. Execute any **[Tools](tool-use.md){ data-preview }** or interact with any interactive **[Attachments](attachments.md){ data-preview }** as needed.
3. Check-off or complete a **Condition (Goal)** or/and return a **[Target](targets.md#targets){ data-preview }** value.

## What are the different Semantic Operations?

The following cards illustrate the different semantic operations provided by `ZYX`.

<div class="grid cards" markdown>

- :lucide-sparkle: **Make**

    Generate outputs based on a **target** type or value.

    [More Info](../semantic-operations/make.md){ data-preview }

- :lucide-goal: **Select**

    Perform a LLM-based selection of one or more options.

    [More Info](../semantic-operations/select.md){ data-preview }

- :lucide-pencil-line: **Edit**

    Edit a **target** value, using automatically determined and type-specific editing strategies.

    [More Info](../semantic-operations/edit.md){ data-preview }

- :lucide-book: **Query**

    Query a grounded **source** object or value, returning a **target** value with confidence and accuracy scores.

    [More Info](../semantic-operations/query.md){ data-preview }

- :lucide-list-check: **Expressions**

    Run a Pythonic expression (`==`, `if x in y`, etc.) against some context or a **source** object or value.

    [More Info](../semantic-operations/expressions.md){ data-preview }

- :lucide-code: **Parse**

    Parse a **source** object or value into a **target** result.

    [More Info](../semantic-operations/parse.md){ data-preview }

- :lucide-check: **Validate**

    Parse a **source** object or value into a **target** result, as well as validating
    the result against a set of constraints.

    [More Info](../semantic-operations/validate.md){ data-preview }

- :lucide-play: **Run**

    *Run* an agent or model on an arbitrary task, and optionally return a **target** value.

    [More Info](../semantic-operations/run.md){ data-preview }

</div>

## Common Parameters

Most semantic operations share a consistent set of parameters. These three are especially useful when you want richer runtime behavior.

### `confidence`

Enable log-probability based confidence scoring on the returned `Result`. This only works for models that expose log-probabilities.

```python title="Confidence Scores"
from zyx import parse
from pydantic import BaseModel


class Info(BaseModel):
    title: str


result = parse(
    source="The Hobbit is a novel by J. R. R. Tolkien.",
    target=Info,
    confidence=True,
)

print(result.confidence)
```

### `observe`

Enable CLI observation of tool calls and progress. Pass `True` for the default observer, or provide a custom `Observer` instance.

```python title="Observation Output"
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
    observe=True,
)
"""
╭─ ▶ Operation ─╮
│ Edit          │
╰───────────────╯
  ⟳ Processing (edit)...
╭─ ✨ Fields Generated ─╮
│   • name              │
╰───────────────────────╯
✓ Edit complete
"""
```

### `stream`

Stream results as they are generated. When enabled, the operation returns a `Stream[T]` instead of a `Result[T]`.

```python title="Streaming Output"
from zyx import query

stream = query(
    source="The Sun is a star at the center of the Solar System.",
    target=str,
    context="Return the key fact in one line.",
    stream=True,
)

for chunk in stream.text():
    print(chunk)

final_result = stream.finish()
print(final_result.output)
```
