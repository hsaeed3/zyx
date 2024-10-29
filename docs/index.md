# `zyx`

---

### An Easier Way to Work With LLMs

`zyx` is a python package built with the sole purpose of providing an intutive
and simple API when working and building with LLMs. It aims to remove the
current boilerplate of most LLM libraries, through *quick-use* functions and
modules.


**The point of `zyx` is not to be a flashy new library, everything here can be
done without much effort, it just helps you do them quicker.** `zyx` is a developer
focused library, it's built so your ideas come out easier.


The framework is built off of incredibly well-built libraries, most notably:

- [Instructor](https://github.com/jxnl/instructor)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [Chroma](https://github.com/chroma-core/chroma)
- [Pydantic](https://github.com/pydantic/pydantic)

---

## Quick Start

### Installation

Install `zyx` using pip:

```bash
pip install --upgrade zyx
```

---

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
