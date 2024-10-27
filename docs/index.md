# `zyx` <span style="color:var(--md-accent-fg-color)">//</span> **a shockingly simple llm library**

---

`zyx` is an llm "*framework*" designed to just work. It's a simple, yet powerful and flexible library that
provides a variety of tools to work with LLM's in a way that has not been this easily possible in other
libraries and frameworks.

Although `zyx` is built to be unobstrusive and as close close to possible as the
standard API for working with LLM's, it does contain a lot of opinionation to help it reach this goal.

<br/>

#### `zyx` provides the following features:

<div class="grid cards" markdown style="display: flex; flex-direction: column; gap: var(--md-grid-spacing);">

- :fire: **Simple .completion() method** - with automatic tool generation, execution, structured outputs & more.
- :people_hugging: **Multi-Agent Pipelines** - with simple & message thread friendly state management for multiple agents.

</div>

<br/>

### Get Started By Clicking a Box Below!

<div class="grid cards" markdown>

- :brain: __Examples__ - Get started with `zyx` by running simple examples.
- :atom: __Concepts__ - Implementations of Popular Applications (**NotebookLM**, **Perplexity** & **more**.)
- :zap: [__LLM Completions__ with automatic tool execution & structured outputs](./examples/llm-completions/getting-started.md)
- 🗄️ __Pydantic Models__ as "LLMs"
- :robot: __Simple & 'State Friendly' Multi Agent Pipelines__ through `zyx.Completions`
- :sparkles: __Magic LLM Functions__ for *classification*, *code generation*, *validation*, *extraction* & more.

</div>

---

### **Quick-Start**

Install `zyx` with pip:

```bash
pip install zyx
```

Run the following command in your terminal for a quick overview in the `zyx` CLI methods:

```bash
zyx

# or
# zyx --help
```

---

### **Example** <span style="color:var(--md-accent-fg-color)">//</span> LLM Completions with Tool Generation

Once you've installed the library try running the following code in your Python environment. This following
snippet showcases the `zyx.completion()` function to create a completion with **automatic tool generation &
execution**.

```python
import zyx as z

# create a completion with automatic tool generation
# passing tools as strings utilizes the `zyx.code()` module; which
# safely generates & executes code in a sandboxed environment.
completion = z.completion(

    messages = "What OS is my system currently running?",   # messages can be passed as a list or a string
                                                            # for ease of use.

    model = "gpt-4o-mini",                                  # the model to use for completion
                                                            # all litellm models are supported

    tools = ["run_cli_command"]                             # lets define a string tool for generation!
                                                            # tools can also be passed as python functions,
                                                            # pydantic models or the standard OpenAI dictionary.

    run_tools = True                                        # execute tool calls automatically
                                                            # defaults to True
)
```

```bash
# OUTPUT
Your system is currently running macOS, specifically with the Darwin kernel version 23.6.0.
```

---

### **Example II** <span style="color:var(--md-accent-fg-color)">//</span> Code Reviewer Agent

```python
# code reviewer agent using zyx agents

import zyx as z

# lets write some code!
my_code = """
def calculate_average(numbers):
    total = 0
    count = 0

    for num in numbers
        if type(num) == 'int':
            total += num
            count += 1
        elif type(num) == 'float'
            total = total + num
            count += 1

    return total/count
"""

# lets create an object to store our code in
class Code(z.BaseModel):
    code : str

# code
code = Code(code = my_code)

# -------------------------

# now we can create an agent to review the code
# initialize completions client
client = z.Completions()

# initialize agents
with client.agents() as agents:

    # initialize task
    with agents.task(object = code) as task:

        # lets review the code utilizing zyx.validate(); an implementation of
        # many research backed methods for LLM guardrails & validation.
        response = task.validate(
            message = "Is the code correct?",
            process = "validate"
        )

        # now lets see if we need to fix the code
        # we will use the 'is_valid' field returned by our validation result
        # since 'is_valid' is boolean, this is a simple conditional! :)
        if not response.is_valid:

            # .patch() is a new method in zyx v1.1.0; that brings in even more
            # control over basemodel generation.
            object = task.patch(
                instructions = "According to the validation & explanation, fix the object."
            )

            # lets validate the code again
            response = task.validate(
                message = "Is the code correct?",
                process = "validate"
            )

        else:
            print("The code is correct!")
```
