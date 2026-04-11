---
title: Results & Streaming
icon: lucide/check-circle
---

# Results & Streaming

Every semantic operation returns either a `Result[T]` or a `Stream[T]`.

## Result

A `Result` is the completed output of an operation along with useful metadata.

```python title="Using Result"
from zyx import make

result = make(target=str, context="Write a one sentence summary of SQL.")

print(result.output)
print(result.model)
print(result.usage)
```

Common fields:

- `output`: the final typed output.
- `raw`: the underlying run results from the model/agent.
- `model`: the model name used for the first run.
- `usage`: total token usage across all runs.
- `all_new_messages`: messages added by the agent during the operation.

### Confidence

If you run with `confidence=True`, the result can compute log-probability based scores (when the model supports it).

```python title="Confidence Scores"
from zyx import make

result = make(
    target=str,
    context="Define entropy in one sentence.",
    confidence=True,
)

print(result.confidence)
```

If confidence is not supported by the model, `result.confidence` may be `None`.

### Semantic Messages

`Result.as_message()` returns a concise semantic summary for context updates. It uses the output value for most operations, and a specialized summary for edit and run operations.

```python title="Semantic Message"
from zyx import make

result = make(target=str, context="Give a short tagline for a bike shop.")

print(result.as_message())
```

## Stream

When `stream=True`, operations return a `Stream[T]` that lets you watch outputs as they are produced.

```python title="Streaming Text"
from zyx import make

stream = make(
    target=str,
    context="Write a short weather poem.",
    stream=True,
)

for chunk in stream.text():
    print(chunk)

final_result = stream.finish()
print(final_result.output)
```

Streaming helpers:

- `text()`: stream text output.
- `partial()`: stream partial structured output.
- `field(name)`: stream a single field from structured output.
- `finish()` / `finish_async()`: finalize and return a `Result`.
- `result`: available after the stream completes.
