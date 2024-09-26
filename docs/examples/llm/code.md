# **Generative Code Functions**

<code>zyx</code> comes with a set of functions to help you generate and execute code, making it easier to get things done in a weirder way. This builds off of the concept provided by Marvin's <code>.fn()</code> function.

## **.function()**

> Create a generative function

```python
from zyx import function

# .function() is a decorator, that generates your defined functions
@function()
def create_pandas_df(data: list[dict]):
    """
    Creates a pandas dataframe from a list of dictionaries.
    """
    
data = [
    {"name": "John", "age": 20},
    {"name": "Jane", "age": 21},
    {"name": "Doe", "age": 22},
]

df = create_pandas_df(data)

print(df.head())
```

```bash
# OUTPUT
   name  age
0  John   20
1  Jane   21
2   Doe   22
```

::: zyx.lib.completions.resources.function.function

## **.code()**

> Directly create a code object

```python
# .code() is an easier to use version of .function()
from zyx import code

logger = code("A logger named zyx logger")

logger.info("Hello, world!")
```

```bash
2024-09-07 23:57:43,792 - zyx_logger - INFO - Hello, world!
```

::: zyx.lib.completions.resources.code.code



