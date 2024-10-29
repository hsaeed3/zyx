# LLM Completions

### Standard LLM Completions

The base `completion()` function runs the `zyx` library, and provides a simple
API for generating

- LLM completions for *any LiteLLM compatible model*
- *Structured outputs* using Pydantic models
- *Tool Calling* & *Automatic Tool Execution*

```python hl_lines="16"
import zyx

def my_favorite_food(name : str) -> str:
    """Returns the user's favorite food."""
    return "pizza"

zyx.completion(
    "What is my favorite food?",
    model = "gpt-4o-mini", # Any LiteLLM model is compatible
    tools = [my_favorite_food], # Any function, basemodel or openai tool
    run_tools = True # Optional: automatically executes tools
    # response_model = BaseModel # Optional: Pydantic model for structured outputs
    # mode = "tool_call" # Optional: Instructor completion mode (refer to instructor docs)
)

# "Your favorite food is pizza!"
```

---

### Passing Messages

The `completion()` function can pass messages as both a dictionary &
a string.

```python
zyx.completion(
    "Hello"
)

# or

zyx.completion(
    messages = [
        { "role" : "system", "message" : "You are a helpful assistant." },
        { "role" : "user", "message" : "Hello" }
    ]
)
```

## API Reference - `completion()`

::: zyx.client.completion