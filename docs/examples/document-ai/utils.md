# Document AI Utilities

`zyx` provides a couple of utilities for working with documents & long text a little easier.

---

## Chunking

Utilize the `chunk` function for quick semantic chunking, with optional parallelization.

```python
from zyx import chunk

chunk("Hello, world!")
# ["Hello, world!"]
```

### API Reference

::: zyx.resources.data.chunk.chunk


---

## Reading

Utilize the `read` function for quick reading of most document types from both local file systems & the web.
Able to injest many documents at once, and return a list of `Document` models.

```python
from zyx import read

read("path/to/file.pdf")
# Document(content="...", metadata={"file_name": "file.pdf", "file_type": "application/pdf", "file_size": 123456})
```

### API Reference

::: zyx.resources.data.reader.read

