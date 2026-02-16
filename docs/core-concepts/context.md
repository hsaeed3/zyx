---
title: Context & Prompting
icon: lucide/layers
---

# Context & Prompting

The `context` parameter is the *memory* and *prompting surface* of `ZYX`. It accepts flexible input types, mixes them freely, and gets normalized into model messages.

At a high level:

1. You pass `context` to any semantic operation.
2. `ZYX` parses and normalizes it into messages.
3. Those messages become the prompt for the model.

## Passing Context

You can pass a single item or a list. Every item is converted to a message under the hood.

### Strings

Strings are treated as user messages.

```python title="String Context"
from zyx import make

result = make(
    target=str,
    context="What is Python?",
)

print(result.output)
"""
Python is a high-level, interpreted programming language known for its simplicity and readability...
"""
```

### Role-Tagged Strings

Use role tags to express multiple messages in one string:

```python title="Role-Tagged Context"
from zyx import make

context = """
[s]You are a helpful assistant.[/s]
[u]What is Python?[/u]
[a]Python is a programming language.[/a]
[u]Tell me more.[/u]
"""

result = make(target=str, context=context)
```

Supported tags: `[s]` / `[system]`, `[u]` / `[user]`, `[a]` / `[assistant]`.

### Message Dicts

OpenAI-style dicts are accepted:

```python title="OpenAI-Style Dicts"
from zyx import make

result = make(
    target=str,
    context=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ],
)
from zyx import make

context = """
[s]You are a helpful assistant.[/s]
[u]What is Python?[/u]
[a]Python is a programming language.[/a]
[u]Tell me more.[/u]
"""

result = make(target=str, context=context)

print(result.output)
"""
Certainly! Python is a high-level, interpreted programming language known for its clear syntax, readability, and versatility. Here are some key features and aspects of Python:...
"""
```

### Snippets

Snippets are perfect for files, URLs, and structured objects:

```python title="Snippets in Context"
from zyx import parse, paste

result = parse(
    source=paste("data.json"),
    target=dict,
    context=[paste("schema.txt"), "Parse according to this schema."],
)
```

### Mix Everything

The best part: you can *mix and match*.

```python title="Mixed Context"
from pydantic_ai import ModelRequest
from zyx import parse, paste, create_context

ctx = create_context(instructions="You are a data analyst")

result = parse(
    source=paste("data.csv"),
    target=dict,
    context=[
        ctx,
        "[s]Follow the rules below.[/s]",
        paste("instructions.md"),
        {"role": "system", "content": "Be thorough."},
        ModelRequest.user_text_prompt("Parse this CSV."),
        paste("reference.json"),
        "[u]Focus on accuracy over speed.[/u]",
        "Proceed with the parsing.",
    ],
)
```

## Context Objects

When you want durable conversation state across operations, use `Context`.

```python title="Using Context"
from zyx import create_context, make

ctx = create_context(
    instructions="You are a helpful assistant.",
    update=True,
)

result1 = make(target=str, context=[ctx, "What is Python?"])
result2 = make(target=str, context=[ctx, "Tell me more about that."])
```

When `update=True` (default), the `Context` automatically stores the full message exchange after each operation.

### Per-Call Overrides

`Context` is callable: it returns a copy with overrides.

```python title="Context Overrides"
from zyx import create_context, make

ctx = create_context(instructions="You are a helpful assistant.")

result = make(
    target=str,
    context=[ctx(instructions="Be concise."), "Summarize this."],
)
```

## Instructions

`Context.instructions` becomes system-level prompts.

```python title="System Instructions"
from zyx import create_context, make

ctx = create_context(instructions="Speak like a pirate.")

result = make(target=str, context=[ctx, "Explain gravity."])
```

If you need structured, multi-line guidance, use double newlines to split into multiple system parts, or set `compact_instructions=True` to keep it as one block.

## Tools and Deps

`Context` can also carry tools and deps:

```python title="Context Tools + Deps"
from zyx import create_context, make

def log_it(text: str) -> None:
    print(text)

ctx = create_context(
    tools=[log_it],
    deps={"trace_id": "abc-123"},
)

result = make(target=str, context=[ctx, "Say hello."])
```

## Excluding Parts of Context

You can selectively disable what gets forwarded:

- `exclude_messages=True`
- `exclude_instructions=True`
- `exclude_tools=True`

```python title="Exclude Context Elements"
from zyx import create_context, make

ctx = create_context(
    instructions="You are a strict assistant.",
    exclude_instructions=True,
)

result = make(target=str, context=[ctx, "What is 2+2?"])
```

## Truncation

If your context grows too long, cap it:

```python title="Limit Message History"
from zyx import create_context, make

ctx = create_context(max_length=5)

result = make(target=str, context=[ctx, "Summarize the conversation."])
```

Only the last `max_length` messages are forwarded. Instructions are still included.

## Manual Message Control

`Context` also exposes helpers when you need to manage messages directly:

- `add_user_message(...)`
- `add_assistant_message(...)`
- `add_system_message(...)`
- `clear()`
- `extend_messages(...)`

Use these sparingly. Most of the time, just pass `context` into your operations and let `ZYX` handle the rest.
