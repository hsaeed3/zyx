---
title: Semantic Operations
icon: lucide/sparkles
---

# Semantic Operations

## What are Semantic Operations?

The core generative, or LLM-powered methods within `ZYX` are called **semantic operations**. A semantic operation is a function that leverages language models to:

1. Take in some input **[Context](context.md){ data-preview }** (prompts, instructions, message history, data) and a **[Source](targets.md){ data-preview }** for specific operations.
2. Execute any **[Tools](tool-use.md){ data-preview }** or interact with any **[Resources](resources.md){ data-preview }** as needed.
3. Check-off or complete a **Condition (Goal)** or/and return a **Target** value.

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

</div>
