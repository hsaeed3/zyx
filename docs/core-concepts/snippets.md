---
title: Snippets
icon: lucide/image
---

# Snippets

A *Snippet* is a piece of content that can be used to represent most Pythonic objects, text, or external content such as web pages, documents, images, audio, video as content that can be easily passed to a model through the `context` or `source` parameters of a semantic operation.

## Types of Snippets

The easiest way to create a *Snippet* is through the `paste()` function, which is able to auto-infer the type of it's input source or object and create a *Snippet* from it.

### HTML & Web Pages

```python title="Using paste() on a URL"
from zyx import paste

snippet = paste("www.google.com") # (1)!

print(snippet.text)
```

1. **Snippets** depend on the `markitdown` library for various text based sources such as HTML, PDFs, Markdown, etc. for quick and accurate rendering to text that can be passed ot a model.

### Multimodal Content

You can also pass in multimodal content such as images, audio, video, documents, etc. from a local file or URL!

```python title="Using paste() on a multimodal content"
from zyx import paste

snippet = paste("https://picsum.photos/id/237/536/354")

print(snippet.text)
"""
[image content at https://picsum.photos/id/237/536/354]
"""
```

### Pythonic Objects

You can also pass in standard Pythonic objects to represent them in the `TOON` (Token-Oriented Object Notation) format that can be cleanly interpreted by a model.

```python title="Using paste() on a Pythonic object"
from zyx import paste

snippet = paste({"name": "John", "age": 30}) # (1)!

print(snippet.text)
"""
name: John
age: 30
"""
```

## Passing Snippets to Semantic Operations

A snippet can be included in the `context` of any semantic operation, or as the direct `source` for operations such as `parse` or `query` that support the parameter.

```python title="Passing a Snippet to a semantic operation"
from zyx import make


snippet = paste("https://picsum.photos/id/237/536/354")


result = make(
    context=[snippet, "What is this image?"], # (1)!
)


print(result.output)
"""
The image shows a black puppy sitting on a wooden surface, looking up at the camera. Its large, expressive eyes and shiny coat suggest it's a young and playful dog.
"""
```

1. A snippet can be included just like any other context item!
