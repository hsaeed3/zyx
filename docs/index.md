---
icon: lucide/book
title: ""
hide:
  - title
---

<span class="page-index"></span>

![ZYX Hero](./assets/zyx-light.png){ align=center .hero-light }
![ZYX Hero](./assets/zyx-dark.png){ align=center .hero-dark }

A fun **"anti-framework"** for doing useful things with agents and LLMs.
{ .hero-tagline }

<p align="center">
<a href="https://pypi.org/project/fastapi" target="_blank">
    <img src="https://img.shields.io/pypi/v/zyx?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
</p>

---

!!! note "What is `ZYX`?"

    `ZYX` is a simplicity-first library wrapped on top of [Pydantic AI]{ data-preview } and heavily inspired by [Marvin]{ data-preview }. It aims to provide a simple, *stdlib-like* interface for working with language models, without loss of control as well as flexibility that more complex frameworks provide.

## Introduction

`ZYX` stands on the shoulders of [Pydantic AI]{ data-preview } to provide a set of functions and components that aim to complete the following goals:

- **Simplicity-First**: Above all else, `ZYX` aims to be as simple as possible to use. The library is designed in a very semantically literal sense, with the hope that a 12 year old could pick up and run with it immediately.
- **Fast & Flexible**: `ZYX` is designed to be as fast as possible in two senses, (1) performance wise, and (2) development time wise, providing a very flexible interface that allows rapid prototyping and iteration.
- **Type-Focused**: `ZYX` provides a very type-focused interface, leveraging [Pydantic]'s powerful validation capabilities to ensure that your data is always in the expected format.
- **Model Agnostic**: Through [Pydantic AI]{ data-preview }, `ZYX` is completely model agnostic and supports virtually *any* LLM provider.

[Pydantic AI]: https://ai.pydantic.dev/ "A Python framework for building production-grade applications with LLMs"
[Pydantic]: https://docs.pydantic.dev/ "A library for data validation and settings management using Python type hints"
[Marvin]: https://askmarvin.ai/ "A clean & simple frameworks for building AI applications"

---

## Quickstart

Here's a few quick examples to get you started with `ZYX`!

??? tip "Installation"

    You can install `ZYX` using your favorite package manager.

    === "pip"

        ```bash
        # install to your current environment
        pip install zyx
        ```

    === "uv"

        ```bash
        # install to your current environment
        uv pip install zyx

        # or add it to your current project
        # uv add zyx
        ```

---

### Generating Content

```python
from zyx import make
from pydantic import BaseModel

class Poem(BaseModel):
    title : str
    verses : list[str]

# generate a poem
# all models supported by Pydantic AI are supported in ZYX!
response = make(
  Poem, context="Generate a poem about the beauty of nature.", model="openai:gpt-4o-mini"
)

print(response.output)
```

```bash title="Output"
Poem(title="Nature's Beauty", verses=["The leaves rustle softly in the breeze,", "A gentle song of leaves and breeze.", "The trees sway in the wind,", "A dance of leaves and breeze."])
```

??? tip "Did you know?"

    The `make` operation is a very versatile operation, and supports passing in no additional context for dynamic synthetic data generation.

    ```python
    response = make(Poem)
    ```

    ??? example "Output"

        ```bash title="Output"
        title='Whispers of Forgotten Seasons' verses=['In the quiet of dusk, shadows intertwine,', 'Leaves whisper secrets, old as the pine.', 'The river hums tales of the days gone by,', 'Where laughter was born, and dreams learned to fly. ', '', 'A flicker of fireflies, dancing in flight,', 'Guide the lost souls through the veil of night.', 'Each twinkle a promise of moments once shared,', 'A tapestry woven, with hearts that once cared.', '', 'The moon, a lone sentinel, gazes with grace,', 'Over fields where the wildflowers once held their place.', 'Their colors now muted, yet fragrant still lingers,', "As nostalgia's soft touch brushes fingers with stingers.", '', 'In the echoes of time, we find our refrain,', 'A melody sweet, woven with joy and with pain.', 'For every season forgotten, a whisper remains,', 'In the heart of the woods, where memory reigns.']
        ```

---

### Using External Sources as Context

```python
from zyx import query, paste

# `query` is a semantic operation for querying against a `source`
# a source can be anything!
# in this case we're linking it to these docs directly
response = query(
  source=paste("https://zyx.hammad.app"),
  context="What library is this built on top of?"
)

print(response.output)
```

```bash title="Output"
Pydantic AI
```