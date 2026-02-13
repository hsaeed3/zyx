# zyx

[![PyPI version](https://img.shields.io/pypi/v/zyx.svg)](https://pypi.org/project/zyx/)
[![Python version](https://img.shields.io/pypi/pyversions/zyx.svg?cacheSeconds=3600)](https://pypi.org/project/zyx/)
[![License](https://img.shields.io/github/license/hsaeed3/zyx.svg)](https://github.com/hsaeed3/zyx/blob/main/LICENSE)

![ZYX](./docs/assets/zyx-dark.png)

> A fun **"anti-framework"** for doing useful things with agents and LLMs.

---

**Documentation**: [https://zyx.hammad.app](https://zyx.hammad.app)

---

`ZYX` is a simplicity-first library wrapped on top of [Pydantic AI](https://ai.pydantic.dev/) and heavily inspired by [Marvin](https://askmarvin.ai/). It aims to provide a simple, *stdlib-like* interface for working with language models, without loss of control or flexibility that more complex frameworks provide.

## Key Features

* **Simplicity-First**: Designed to be as simple as possible to use. The library uses semantically literal function names—a 12-year-old could pick it up and run with it immediately.
* **Fast & Flexible**: Optimized for both performance and development speed, providing a flexible interface that enables rapid prototyping and iteration.
* **Type-Focused**: Leverages [Pydantic](https://docs.pydantic.dev/)'s powerful validation capabilities to ensure your data is always in the expected format, with full type safety and editor support.
* **Model Agnostic**: Through [Pydantic AI](https://ai.pydantic.dev/), `ZYX` supports virtually *any* LLM provider—OpenAI, Anthropic, Google, and more.
* **Semantic Operations**: A clean set of operations (`make`, `parse`, `query`, `edit`, `select`, `validate`) that cover common LLM use cases.
* **Resource Management**: Built-in support for files, code, and vector memory as first-class resources.
* **Streaming Support**: Built-in streaming for real-time responses and partial results.

## Requirements

`ZYX` stands on the shoulders of giants:

* [Pydantic AI](https://ai.pydantic.dev/) for LLM integration
* [Pydantic](https://docs.pydantic.dev/) for data validation
* [MarkItDown](https://github.com/microsoft/markitdown) used by `Snippets` for rendering to text
* Python 3.11+

## Installation

Install `ZYX` using pip:

```bash
pip install zyx
```

By default, `ZYX` installs the minimum required dependencies from [Pydantic AI](https://ai.pydantic.dev/) using their `pydantic-ai-slim` package, along with the `openai` library for out-of-the-box OpenAI support.

### Additional Providers

To add support for additional LLM providers, you can either install the entire `pydantic-ai` package or add providers individually:

```bash
# Install all providers
pip install zyx pydantic-ai

# Or use the `ai` extra
pip install 'zyx[ai]'

# Or add providers manually
pip install zyx anthropic
```

## Quickstart

### Generate Content

The easiest way to use LLMs within `ZYX` is through **semantic operations**. Start with `zyx.make` to generate content:

```python
import zyx

result = zyx.make(
    target=int,
    context="What is 45+45?",
    model="openai:gpt-4o-mini",
)

print(result.output)
# 90
```

### Parse Structured Data

Use `zyx.parse` to extract structured data from any source:

```python
import zyx
from pydantic import BaseModel

class Information(BaseModel):
    library_name: str
    library_description: str

result = zyx.parse(
    source=zyx.paste("https://zyx.hammad.app"),
    target=Information,
    model="openai:gpt-4o-mini",
)

print(result.output.library_name)
# ZYX
```

### Query with Tools

Add tools to any semantic operation for enhanced capabilities. Notice how we mix a snippet, string, and OpenAI dict:

```python
import zyx
from pydantic import BaseModel

class Information(BaseModel):
    library_name: str
    library_description: str

def log_website_url(url: str) -> None:
    print(f"Website URL: {url}")

result = zyx.parse(
    source=zyx.paste("https://zyx.hammad.app"),
    target=Information,
    context=[
        {"role": "system", "content": "You are a web scraper."},
        zyx.paste("scraping_instructions.txt"),
        "log the website URL before you parse.",
    ],
    model="openai:gpt-4o-mini",
    tools=[log_website_url],
)

print(result.output.library_name)
print(result.output.library_description)
```

### Edit Values

Use `zyx.edit` to modify existing values:

```python
import zyx

data = {"name": "John", "age": 30}

result = zyx.edit(
    target=data,
    context="Update the age to 31",
    model="openai:gpt-4o-mini",
)

print(result.output)
# {"name": "John", "age": 31}
```

### Query Grounded Sources

Use `zyx.query` to ask questions about a source:

```python
import zyx

result = zyx.query(
    source="Python is a high-level programming language...",
    target=str,
    context="What is Python?",
    model="openai:gpt-4o-mini",
)

print(result.output)
```

### Select from Options

Use `zyx.select` to choose from a list of options:

```python
import zyx
from typing import Literal

Color = Literal["red", "green", "blue"]

result = zyx.select(
    target=Color,
    context="What color is the sky?",
    model="openai:gpt-4o-mini",
)

print(result.output)
# blue
```

### Async Support

All operations have async variants:

```python
import zyx

result = await zyx.amake(
    target=str,
    context="Write a haiku about Python",
    model="openai:gpt-4o-mini",
)

print(result.output)
```

### Streaming

Get real-time results with streaming:

```python
import zyx

stream = zyx.make(
    target=str,
    context="Write a short story",
    model="openai:gpt-4o-mini",
    stream=True,
)

for text in stream.text():
    print(text, end="", flush=True)
```

## Semantic Operations

`ZYX` provides a set of semantic operations that cover common LLM use cases:

* **`make`** / **`amake`**: Generate new content for a target type or value
* **`parse`** / **`aparse`**: Extract structured content from a primary source
* **`query`** / **`aquery`**: Answer questions grounded in the primary source
* **`edit`** / **`aedit`**: Modify a target value or object, optionally selective or planned
* **`select`** / **`aselect`**: Choose from a list/enum/literal set
* **`validate`** / **`avalidate`**: Parse then verify constraints; can return structured violations
* **`expr`**: Semantic expression helpers (`==`, `in`, `bool`) using `parse`

## Resources

`ZYX` provides first-class support for resources that models can read, query, or mutate:

* **`File`**: Filesystem files (text/JSON/YAML/TOML/etc.) with anchor-based edits
* **`Code`**: Code files with language-aware parsing and editing
* **`Memory`**: Vector-store-backed resource for add/search/delete operations

## Context Management

The `context` parameter in semantic operations accepts flexible input types that are automatically converted to messages. **You can mix and match any combination of context types however you want**—pass a single item or a list of items, and they'll be combined into a conversation history.

### String Context

Simple strings are treated as user messages:

```python
import zyx

result = zyx.make(target=str, context="What is Python?")
```

### Role-Tagged Strings

Use role tags to create multiple messages in a single string:

```python
import zyx

# Supported tags: [s]/[system], [u]/[user], [a]/[assistant]
context = """
[s]You are a helpful assistant.[/s]
[u]What is Python?[/u]
[a]Python is a programming language.[/a]
[u]Tell me more.[/u]
"""

result = zyx.make(target=str, context=context)
```

### Context Objects

Use `zyx.Context` to manage conversation history, instructions, and tools across operations:

```python
import zyx

ctx = zyx.create_context(
    instructions="You are a helpful assistant",
    auto_update=True,
)

# Pass context as a list item
result1 = zyx.make(target=str, context=[ctx, "What is Python?"])
result2 = zyx.make(target=str, context=[ctx, "Tell me more about that"])
```

### Snippets

Use `zyx.paste()` to include files, URLs, or other content as snippets:

```python
import zyx

# Include a file
result = zyx.parse(
    source=zyx.paste("data.json"),
    target=dict,
    context=[zyx.paste("schema.txt"), "Parse this according to the schema"],
)

# Include a URL
result = zyx.query(
    source=zyx.paste("https://example.com/article"),
    target=str,
    context="Summarize this article",
)

# Include raw bytes or objects
result = zyx.parse(
    source=zyx.paste(b"raw binary data"),
    target=str,
    context="What is this?",
)
```

### Mix and Match Freely

**The real power is mixing everything together.** Combine strings, Context objects, snippets, OpenAI dicts, role-tagged strings, and more—however you want:

```python
import zyx

ctx = zyx.create_context(instructions="You are a data analyst")

# Wild combination example - mix everything!
result = zyx.parse(
    source=zyx.paste("data.csv"),
    target=dict,
    context=[
        # Start with a Context object
        ctx,
        # Add a role-tagged string
        "[s]Also consider the following guidelines.[/s]",
        # Include a file snippet
        zyx.paste("instructions.md"),
        # Add an OpenAI-style system message
        {"role": "system", "content": "Be thorough in your analysis."},
        # Mix in a simple string
        "Parse the CSV file according to all the instructions above",
        # Add more snippets
        zyx.paste("reference_data.json"),
        # Another role-tagged message
        "[u]Focus on accuracy over speed.[/u]",
        # More OpenAI dicts
        {
            "role": "assistant",
            "content": "I understand. I'll parse the data carefully."
        },
        # Final instruction as plain string
        "Now proceed with the parsing.",
    ],
)

# Another example: building context dynamically
messages = [
    {"role": "system", "content": "You are a code reviewer."},
    zyx.paste("code.py"),
    "[s]Review this code for security issues.[/s]",
    zyx.paste("security_guidelines.md"),
    "Provide detailed feedback.",
]

result = zyx.query(
    source=zyx.paste("code.py"),
    target=str,
    context=messages,
)
```

### Message Dictionaries

Pass OpenAI-style message dictionaries or PydanticAI ModelMessage dictionaries:

```python
import zyx

# OpenAI-style user message
user_msg = {
    "role": "user",
    "content": "Hello!"
}

# OpenAI-style system message
system_msg = {
    "role": "system",
    "content": "You are a helpful assistant."
}

# OpenAI-style assistant message
assistant_msg = {
    "role": "assistant",
    "content": "Hello! How can I help you?"
}

# OpenAI-style multimodal message (with images)
multimodal_msg = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    ]
}

# OpenAI-style tool message
tool_msg = {
    "role": "tool",
    "name": "get_weather",
    "content": "Sunny, 72°F"
}

# Mix OpenAI dicts with other context types
result = zyx.make(
    target=str,
    context=[
        system_msg,
        user_msg,
        assistant_msg,
        "Continue the conversation",
        zyx.paste("additional_context.txt"),
    ]
)

# PydanticAI format
pydantic_ai_message = {
    "parts": [{"type": "user_prompt", "content": "Hello!"}]
}

result = zyx.make(target=str, context=[pydantic_ai_message])
```

### Pydantic Models

Pydantic models are automatically converted to dictionaries:

```python
import zyx
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

msg = Message(role="user", content="Hello!")

result = zyx.make(target=str, context=[msg])
```

## Type Safety

`ZYX` leverages Python's type system and Pydantic for full type safety:

```python
import zyx
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

result = zyx.parse(
    source='{"name": "Alice", "age": 30}',
    target=User,
    model="openai:gpt-4o-mini",
)

# result.output is fully typed as User
print(result.output.name)  # Type-checked!
print(result.output.age)   # Type-checked!
```

## Documentation

For more detailed documentation, examples, and guides, visit:

**[https://zyx.hammad.app](https://zyx.hammad.app)**

## License

This project is licensed under the terms of the MIT license.
