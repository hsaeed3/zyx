---
icon: lucide/smile
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
  <a href="https://pypi.org/project/zyx" target="_blank">
    <img src="https://img.shields.io/pypi/v/zyx.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/zyx" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/zyx.svg?cacheSeconds=3600" alt="Python version">
  </a>
  <a href="https://github.com/hsaeed3/zyx/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/hsaeed3/zyx.svg" alt="License">
  </a>
</p>

---

!!! note "What is `ZYX`?"

    `ZYX` is a simplicity-first library wrapped on top of [Pydantic AI]{ data-preview } and heavily inspired by [Marvin]{ data-preview }. It aims to provide a simple, *stdlib-like* interface for working with language models, without loss of control as well as flexibility that more complex frameworks provide.


[Pydantic AI]: https://ai.pydantic.dev/ "A Python framework for building production-grade applications with LLMs"
[Pydantic]: https://docs.pydantic.dev/ "A library for data validation and settings management using Python type hints"
[Marvin]: https://askmarvin.ai/ "A clean & simple frameworks for building AI applications"

## Introduction

`ZYX` stands on the shoulders of [Pydantic AI]{ data-preview } to provide a set of functions and components that aim to complete the following goals:

- **Simplicity-First**: Above all else, `ZYX` aims to be as simple as possible to use. The library is designed in a very semantically literal sense, with the hope that a 12 year old could pick up and run with it immediately.
- **Fast & Flexible**: `ZYX` is designed to be as fast as possible in two senses, (1) performance wise, and (2) development time wise, providing a very flexible interface that allows rapid prototyping and iteration.
- **Type-Focused**: `ZYX` provides a very type-focused interface, leveraging [Pydantic]{ data-preview }'s powerful validation capabilities to ensure that your data is always in the expected format.
- **Model Agnostic**: Through [Pydantic AI]{ data-preview }, `ZYX` is completely model agnostic and supports virtually *any* LLM provider.

---

## Getting Started

The following guide is a quick way to get started on how to
install `ZYX` and get up and running with a few of the core concepts.

### Installation

You can install `ZYX` using your favorite Python package manager.

=== "pip"

    ```bash
    pip install zyx
    ```

=== "uv"

    ```bash
    uv add zyx
    ```

=== "poetry"

    ```bash
    poetry add zyx
    ```

=== "conda"

    ```bash
    conda install zyx
    ```

??? tip "Adding Model Providers"

    The core package for `ZYX` installs the minimum required dependencies from [Pydantic AI] using their `pydantic-ai-slim` package, along with the `openai` library for out of the box OpenAI support.

    To add support for additional LLM providers, you can use one of the following methods to install the relevant dependencies:

    === "Install All Providers at Once"

        To install all of `pydantic-ai`'s supported providers, simply install the full package along with `ZYX`:

        ```bash
        pip install zyx pydantic-ai
        ```

        Or use the `ai` extra:

        ```bash
        pip install 'zyx[ai]'
        ```

    === "Install Specific Providers"

        If you dont want to install the entire `pydantic-ai` package, you can easily add your desired providers individually, either through installing their libraries directly, or through a `ZYX` extra:

        ```bash
        # install provider libraries directly
        pip install zyx anthropic
        ```

---

## Quickstart

Once you've installed `ZYX`, try running some of the following code examples to get a feel for the library!

### Generating Content

The easiest way to use LLMs within `ZYX` is through the `zyx.make` **semantic operation**. Semantic operations are specialized functions that allow you to use LLMs to generate/edit content, parse data, and more!

```python title="Generating Content"
import zyx


result = zyx.make(
    target=int, # (1)!
    context="What is 45+45?",
)


print(result.output)
"""
90
"""
```

1. **`Targets`** are how to define the output of a semantic operation. In this case, we want to generate an integer.

### Parsing a Website

Through the **`source`**, **`context`** and **`attachments`** parameters, `ZYX` provides a bunch of options on how you want to pass content to a model.

```python title="Parsing a Website"
import zyx
from pydantic import BaseModel


class Information(BaseModel): # (1)!
    library_name : str
    library_description : str


def log_website_url(url : str) -> None:
    print(f"Website URL: {url}")


result = zyx.parse( # (2)!
    source=zyx.paste("https://zyx.hammad.app"), # (3)!
    target=Information,
    context="log the website URL before you parse.", # (4)!
    model="openai:gpt-4o-mini", # (5)!
    tools=[log_website_url] # (6)!
)


print(result.output.library_name)
print(result.output.library_description)
"""
Website URL: https://zyx.hammad.app
ZYX
A fun "anti-framework" for doing useful things with agents and LLMs.
"""
```

1. First-class support for structured outputs in nearly every semantic operation within `ZYX`.

2. `zyx.parse` is a **semantic operation** that allows you to parse a `source` object or content into a `target`.

3. `zyx.paste` is a convenience function for creating a **`Snippet`** from a source object. <br/> A **`Snippet`** is a piece of content that can be used to represent multimodal or textual content easily.

4. The `context` parameter is how prompts and messages are provided to the model. `ZYX` provides many options on how context can be provided!

5. All models and providers supported by [Pydantic AI]{ data-preview } are supported within `ZYX`.

6. Thanks to [Pydantic AI]{ data-preview }, all **semantic operations** inherently support tool usage.
