# Label Based Classification

The `classify()` function is used to classify text inputs based on a set of labels. The function can be used for both single label and multi label classification.

---

### Single Label Classification

```python
import zyx

inputs = [
    "I love programming in Python",
    "I like french fries"
]

labels = [
    "code",
    "food"
]

result = zyx.classify(
    inputs = inputs,
    labels = labels,
)

print(result)
```

```bash
ClassificationResult(text='I love programming in Python', label='code')
```

---

### Multi Label Classification

```python
from zyx import classify

inputs = [
    "I love programming in Python and food",
    "I like french fries"
]

labels = [
    "code",
    "food"
]

result = classify(
    inputs = inputs,
    labels = labels,
    classification="multi",
    model="openai/gpt-4o",
)

print(result)
```

```bash
MultiClassificationResult(text='I love programming in Python and food', labels=['code', 'food']),
MultiClassificationResult(text='I like french fries', labels=['food'])
```

---

## API Reference

::: zyx.resources.completions.base.classify.classify
