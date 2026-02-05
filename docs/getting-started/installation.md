---
icon: lucide/zap
title: "Getting Started"
---

# Installation

If you've made it this far, it's fair to say you're interested in this library, and want to get up and running as soon as possible. This guide will walk you through the basics of installing `ZYX`, as well as a few *quickstart* examples to get you familiar with the core patterns and components in the library.

You can install the library through `PyPI` using your favorite package manager.

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

## Adding Model Providers

!!! tip "Tip"

    The core package for `ZYX` installs the minimum required dependencies from [Pydantic AI] using their `pydantic-ai-slim` package, along with the `openai` library for out of the box OpenAI support.

    To add support for additional LLM providers, you can use one of the following methods to install the relevant dependencies:

    === "Install All Providers at Once"

        To install all of `pydantic-ai`'s supported providers, simply install the full package along with `ZYX`:
    
        ```bash
        pip install zyx pydantic-ai
        ``` 

    === "Install Specific Providers"

        If you dont want to install the entire `pydantic-ai` package, you can easily add your desired providers individually, either through installing their libraries directly, or through a `ZYX` extra:

        ```bash
        # install provider libraries directly
        pip install zyx anthropic

        # or install through zyx extras
        pip install 'zyx[gemini]'
        ```

## Quickstart Example

Here's a quick example to help you get started with `ZYX`:

```python
from zyx import make

response = make(
    # zyx is very friendly on how you decide to pass context and prompts
    # to your LLMs.
    context="""[s] You are a helpful assistant who only responds in poems. [/s]
    [u] What is the capital of France? [/u]
    """,

    # use any provider supported by Pydantic AI
    model = "openai:gpt-4o-mini",

    # allow the model to respond in a 'target' type
    target = list[str]
)

print(response.output)
```

```bash title="Output"
['In a city where romance sings,', 'The Seine flows softly, love takes wing.', "Paris, the heart of France's embrace,", 'A capital known for its beauty and grace.']
```

[Pydantic AI]: https://ai.pydantic.dev/ "A Python framework for building production-grade applications with LLMs"