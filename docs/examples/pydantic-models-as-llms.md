# **Using Pydantic Models as LLM Clients**

One of the most fun additions I've made to `zyx` is the ability to directly generate & work with `Pydantic` models, through the specialized `BaseModel` provided by the library, that has been deeply integrated with LLM methods.

These methods bring in a new level of integration between LLMs & `Pydantic` models, allowing for a more seamless experience when working with structured data. Some of the power this implementation brings includes:

```

- **Chain of Thought** by seqentially generating Pydantic fields
- Instantly Generating Synthetic Data from Pydantic Models
- Making Patches/Edits to Specific Fields
- Using Content in Pydantic Models as Context for Completions (**RAG!**)
- Generating content directly into a specific Pydantic field
- & more! :)

```

Let's dive into some examples!

<br/>

---

<br/>

---

## **What functions are available?**

All functions available on `BaseModel` are prefixed with `model_`. The available functions are:

- `model_generate()`
- `model_regenerate()`
- `model_completion()`
- `model_patch()`
- `model_select()`

## **Generating Synthetic Data**

### Generating Data from a Pydantic Model

To generate data from a Pydantic model, you can use the `model_generate` method. This method will generate a single instance of the model.

The reason the methods are prefixed with `model_` is because `Pydantic` themselves use `model_` for their methods.

```python
# Generating Data from a Pydantic Model
from zyx import BaseModel

class User(BaseModel):
    name: str
    age: int

User.model_generate()
```

```bash
# Output

User(name='Alice Johnson', age=30)
```

<br/>

### Generating Batch Data 

Using the same method, you can generate a batch of data by passing in the `n` parameter.

```python
# Generating Batch Data 
from zyx import BaseModel

class User(BaseModel):
    name: str
    age: int

# any litellm model will work
print(User.model_generate(model = "gpt-4o-mini", n = 10))
```

```bash
# Output

[
    User(name='Alice', age=30),
    User(name='Bob', age=25),
    User(name='Charlie', age=35),
    User(name='Diana', age=28),
    User(name='Ethan', age=22),
    User(name='Fiona', age=40),
    User(name='George', age=33),
    User(name='Hannah', age=27),
    User(name='Ian', age=29),
    User(name='Julia', age=31)
]
```

<br/>

---

## **Chain of Thought**

The `model_generate` supports a `sequential` generation mode, which allows you to generate data field by field, in a chain of thought manner.

This is useful when you want to generate data that adheres to a specific schema, but where each field is generated based on the previous fields.

```python
# Chain of Thought
from zyx import BaseModel
from typing import Optional

# create some sort of 'reasoning' model
class ComplexSolution(BaseModel):
    equation: str
    process: Optional[str] = None
    reasoning: Optional[str] = None
    answer: Optional[int] = None

# basemodels can be used normally
complex_solution = ComplexSolution(
    equation = "x^2 + 2x + 1"
)

# generate the data
# process is "batch" by default
complex_solution = complex_solution.model_generate(
    model = "gpt-4o-mini", process = "sequential")

print(complex_solution)
```

```bash
# Output

ComplexSolution(
    equation='x^2 + 2x + 1 = 0',
    process='Solving the equation x^2 + 2x + 1 = 0 using the quadratic formula.',
    reasoning='The equation x^2 + 2x + 1 = 0 can be factored as (x + 1)(x + 1) = 0, which gives the solution x = 
-1. This is confirmed by applying the quadratic formula, where a = 1, b = 2, and c = 1.',
    answer=-1
)
```

<br/>

---

## **Using Content in Pydantic Models as Context for Completions (RAG!)**

By utilizing the `model_completion` method, the model can be utilized as prompt context for completions, allowing for easy RAG. 

> Obviously this is not good rag, but it is rag nonetheless

```python
# Using Content in Pydantic Models as Context for Completions (RAG!)
context = [
    "i really like apples",
    "some people like oranges more",
    "bananas are yellow",
    "some people like bananas",
]

# dumb example i know
from zyx import BaseModel
from typing import List, Optional


class Context(BaseModel):
    context : List[str]

# lets fill in our model
context_model = Context(
    context = context,
)

# get the completion
# using .model_completion()
answer = context_model.model_completion("what is my favorite fruit?")

# BaseModel.completion() returns the same `ChatCompletion` instance as zyx.completion()
# lets print the content
print(answer.choices[0].message.content)
```

```bash
# Output

Based on your context, it seems like you really like apples.
```

<br/>

---

## **Selecting Content**

I'm still trying to work out the best implementation for something like this, but this is how to select content

