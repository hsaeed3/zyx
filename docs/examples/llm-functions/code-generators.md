# Code Generators

The `function()` and `code()` functions are code generation & execution modules that can be used for generating working python functions and
objects from a natural language input. The functions provide the ability to return either a mock or real implementation of the response.

---

### Using Generated Code

The code function is used to quickly generate python code from a natural language input. The function can be used to generate code snippets, functions, or even entire classes.

```python
import zyx

logger = zyx.code("A logger named my logger", model = "gpt-3.5-turbo")

logger.info("Hello world!")
```

```bash
2024-09-29 13:17:25,573 - my_logger - INFO - Hello world!
```

---

### Generating Functions

The `function()` decorator is used for slightly more complex use cases; and executes functions themsevles. Functions must be defined with a **docstring**.

For this example lets begin by creating some data.

```python
import zyx

# Lets define some data for this example
data = {
    "Names" : ['John', 'Doe', 'Jane'],
    "Ages" : [23, 45, 32],
    "Locations" : ['USA', 'UK', 'Canada'],
    "Occupations" : ['Doctor', 'Engineer', 'Teacher']
}
```

Now lets define a function that will convert this data into a pandas dataframe

```python
@zyx.function(verbose = True)
def build_names_into_dataframe(data : dict):
    """Creates a proper pandas dataframe from given dictionary data"""


df = build_names_into_dataframe(data)
```

```bash
Names  Ages Locations Occupations
0  John    23       USA      Doctor
1   Doe    45        UK    Engineer
2  Jane    32    Canada     Teacher
```
