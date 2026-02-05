---
title: Semantic Operations
icon: lucide/sparkles
---

# Semantic Operations

## What are Semantic Operations?

The core generative, or LLM-powered methods within `ZYX` are called **semantic operations**. A semantic operation is a function that leverages language models to:

1. Take in some input **[Context](context.md){ data-preview }** (prompts, instructions, message history, data).
2. Execute any **[Tools](tool-use.md){ data-preview }** or interact with any **[Resources](../context/resources.md){ data-preview }** as needed.
3. Check-off or complete a **Condition (Goal)** or/and return a **Target** value.

## What are the different Semantic Operations?

The following cards illustrate the different semantic operations provided by `ZYX`.

<div class="grid cards" markdown>

- :lucide-sparkle: **Make**

    Generate outputs based on a **target** type or value.

    [More Info](../semantic-operations/make.md){ data-preview }

- :lucide-goal: **Run**

    Let an LLM/agent run until it has completed a **condition** and return a **target** value.

    [More Info](../semantic-operations/run.md){ data-preview }

- :lucide-pencil-line: **Edit**

    Edit a **target** value, using automatically determined and type-specific editing strategies.

    [More Info](../semantic-operations/edit.md){ data-preview }

- :lucide-book: **Query**

    Query a grounded **source** object or value, returning a **target** value with confidence and accuracy scores.

    [More Info](../semantic-operations/query.md){ data-preview }

- :lucide-list-check: **Evaluate**

    Run a Pythonic expression (`==`, `if x in y`, etc.) against some context or a **source** object or value.

    [More Info](../semantic-operations/evaluate.md){ data-preview }

- :lucide-code: **Parse**

    Parse a **source** object or value into a **target** result.

    [More Info](../semantic-operations/parse.md){ data-preview }

</div>

## How do I use Semantic Operations?

The easiest method to use a semantic operation is to simply call the function directly. Let's start with a simple
example to generate a structured output.

```python
from zyx import make

response = make(
    # all semantic operations take in some context
    context="What is 2+2?",
    # the `make` operation returns a `target` type
    target=int
)
```

```bash
>>> 4
```