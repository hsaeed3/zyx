# **NLP Functions**

<samp>zyx</samp> comes with a set of NLP functions that are designed to work with text data. All of these functions are inspired from <code>MarvinAI</code>.

## **.classify()**

> Classify text into multiple categories

```python
from zyx import classify

labels = ["positive", "negative", "neutral"]

text = [
    "I love this product!",
    "This is the worst thing I've ever used.",
    "I'm indifferent about this.",
    "What a wonderful day!",
    "I'm not sure about this.",
    "My day was horrible.",
    "Thank you for your help!"
]

classifications = classify(inputs = text, labels = labels)

print(classifications)
```

```bash
# OUTPUT
[
    [ClassificationResult(text='I love this product!', label='positive')],
    [ClassificationResult(text="This is the worst thing I've ever used.", label='negative')],
    [ClassificationResult(text="I'm indifferent about this.", label='neutral')],
    [ClassificationResult(text='What a wonderful day!', label='positive')],
    [ClassificationResult(text="I'm not sure about this.", label='neutral')],
    [ClassificationResult(text='My day was horrible.', label='negative')],
    [ClassificationResult(text='Thank you for your help!', label='positive')]
]
```

::: zyx.lib.completions.resources.classify.classify


## **.extract()**

> Extract information from text into labels

```python
from zyx import extract
from pydantic import BaseModel

text = "One day, John went to the store. He bought a book and a pen. He then went to the park. He played football with his friends. He had a great time."

class Entities(BaseModel):
    names: list[str]
    places: list[str]
    items: list[str]

extracted = extract(Entities, text)

print(extracted)
```

```bash
# OUTPUT
Entities(names=['John'], places=['store', 'park'], items=['book', 'pen', 'football'])
```

::: zyx.lib.completions.resources.extract.extract

## **.generate()**

> Generate pydantic models quickly & efficiently

```python
from zyx import generate
from pydantic import BaseModel

class Book(BaseModel):
    title: str
    author: str
    pages: int

books = generate(Book, n = 10, model = "openai/gpt-4o-mini")

print(books)
```

```bash
# OUTPUT
[
    Book(title='The Great Gatsby', author='F. Scott Fitzgerald', pages=180),
    Book(title='1984', author='George Orwell', pages=328),
    Book(title='To Kill a Mockingbird', author='Harper Lee', pages=281),
    Book(title='Pride and Prejudice', author='Jane Austen', pages=279),
    Book(title='The Catcher in the Rye', author='J.D. Salinger', pages=214),
    Book(title='Moby Dick', author='Herman Melville', pages=585),
    Book(title='War and Peace', author='Leo Tolstoy', pages=1225),
    Book(title='The Hobbit', author='J.R.R. Tolkien', pages=310),
    Book(title='Fahrenheit 451', author='Ray Bradbury', pages=158),
    Book(title='The Alchemist', author='Paulo Coelho', pages=208)
]
```

::: zyx.lib.completions.resources.generate.generate
