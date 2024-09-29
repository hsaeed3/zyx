# Entity Extraction

The `extract()` function can be used to extract any kind of textual information from a given source. As the LLM is the one performing the task,
extraction is semantically aware.

---

## Using the `extract()` function

```python
import zyx

class User(BaseModel):
    name: str
    age: int

zyx.extract(User, "John is 20 years old")
```

```bash
User(name='John', age=20)
```

---

## API Reference

::: zyx.resources.completions.base.extract.extract
