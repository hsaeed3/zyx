---
title: Targets & Sources
icon: lucide/target
---

# Targets & Sources

A **Target** describes the output you want from a semantic operation. It can be a Python type, a concrete value, or a `Target` object with extra metadata and hooks.

A **Source** is the primary input for operations like `parse`, `query`, `edit`, and `validate`. For `make` and `select`, a source is usually optional or implicit.

## Target Basics

Targets can be as simple as a type:

```python title="Using a Type Target"
from zyx import make

result = make(target=str, context="Write a short headline about cats.")

print(result.output)
```

Or a Pydantic model for structured output:

```python title="Structured Target"
from zyx import make
from pydantic import BaseModel


class Article(BaseModel):
    title: str
    summary: str


result = make(
    target=Article,
    context="Write a title and summary about renewable energy.",
)

print(result.output)
"""
title='Harnessing the Future: The Rise of Renewable Energy' summary='This article explores the growing importance of renewable energy sources such as solar, wind, and hydroelectric power in combating climate change and promoting sustainable development. It examines recent advancements in technology, government policies that support the transition to cleaner energy, and the economic benefits of investing in green energy. Additionally, the article highlights success stories from various countries that have made significant strides in reducing their carbon footprints and fostering energy independence through renewable resources.'
"""
```

## Target Metadata

Use `target(...)` to add extra guidance such as a name, description, instructions, constraints, or a default model.

```python title="Target Metadata"
from zyx import target, parse
from pydantic import BaseModel


class Movie(BaseModel):
    title: str
    year: int


movie_target = target(
    Movie,
    name="Movie Info",
    description="Basic movie metadata",
    instructions="Only return the title and release year.",
)

result = parse(
    source="Alien was released in 1979.",
    target=movie_target,
)

print(result.output)
```

## Constraints

Targets can include constraints that `validate` (or `Target.validate`) will check using a model.

```python title="Target Constraints"
from zyx import target
from pydantic import BaseModel


class Task(BaseModel):
    title: str
    priority: str


task_target = target(
    Task,
    constraints=[
        "priority must be one of: low, medium, high",
        "title must be concise",
    ],
)

result = task_target.validate(
    source="Fix the build, urgent.",
)

print(result)
```

## Target Hooks

Targets can register hooks that run on completion or error, or on specific fields as they are produced. Hooks can retry or update the output depending on the hook options.

```python title="Target Hooks"
from zyx import target, make


profile = target(dict, name="Profile")


@profile.on_field("name", retry=True)
def clean_name(value: str) -> str:
    return value.strip().title()


@profile.on("complete")
def on_complete(result):
    return result


result = make(
    target=profile,
    context="name:   jane doe\nlocation: oakland",
)

print(result.output)
```

## Sources

Sources are the primary inputs for grounded operations:

```python title="Parsing a Source"
from zyx import parse
from pydantic import BaseModel


class Info(BaseModel):
    product: str
    price: float


result = parse(
    source="The mug costs $12.50.",
    target=Info,
)

print(result.output)
```

You can pass a `source` as raw text, a Python object, or an attachment created with `paste(...)`.
