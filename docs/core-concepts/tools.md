---
title: Tools
icon: lucide/wrench
---

# Tools

Tools let models and agents call functions during a semantic operation. You can pass tools directly with the `tools` parameter or attach tool-providing resources via `attachments`.

## Passing Tools to Semantic Operations

You can pass any compatible `pydantic_ai` tool to a semantic operation via the `tools` parameter. This includes:

- Functions
- PydanticAI Tools
- PydanticAI Builtin Tools
- PydanticAI Toolsets

```python title="Passing a function as a tool"
from zyx import make
from zyx.tools import Tool

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

result = make(
    target=str,
    context="What is the weather in Tokyo?",
    tools=[get_weather],
)

print(result.output)
"""
The weather in Tokyo is sunny.
"""
```

## Prebuilt Tools

=== "Memory"

    The `Memory` tool is a prebuilt toolset that can be used by a model or agent to store and query memories.

    ```python title="Using Memory as a tool"
    from zyx import make
    from zyx.tools import Memory


    memory = Memory(
        key="project_notes",
        provider="chroma/ephemeral",
        instructions="Stores project notes.",
    )


    memory.add("The project is a new social media platform that allows users to share their thoughts and ideas with others.")


    result = make(
        str,
        context="What is the project notes about in memory?",
        attachments=[memory], # (1)!
    )


    print(result.output)
    """
    The project notes memory stores various notes related to projects. 
    If you would like more specific information or need to search through the project notes, please let me know!
    """
    ```

    1. Various prebuilt tools within `ZYX` can be passed directly as attachments to a semantic operation, for additional context and functionality that is provided to the model or agent.

=== "Code"

    The `Code` tool is a prebuilt toolset that utilizes the new `pydantic_monty` library to provide code execution functionality to a model or agent.

    ```python title="Using Code as a tool"
    from zyx import make
    from zyx.tools import Code


    code = Code(
        script_name="calc.py",
        external_functions={"double": lambda x: x * 2},
    )


    result = make(
        context="Compute double(21) using the code tool and return the answer.",
        tools=[code],
    )


    print(result.output)
    """
    The result of double(21) is 42.
    """
    ```
