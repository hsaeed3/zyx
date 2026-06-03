---
title: Attachments
icon: lucide/image
---

# Attachments

An *Attachment* is a piece of content or an object reference that can be used to represent text, documents, multimodal content, or Python objects in a way that can be passed to a model via the `context`, `source`, or `attachments` parameters of a semantic operation.

There are two ways to create an attachment:

- `paste(...)` creates a static, read-only attachment from a source.
- `attach(...)` creates an interactive attachment that a model can query or edit (when supported by the source).

## Types of Attachments

The easiest way to create an attachment is through the `paste()` function, which auto-infers the type of its input source and renders it into a model-friendly form.

### HTML & Web Pages

```python title="Using paste() on a URL" hl_lines="3"
from zyx import paste

attachment = paste("www.google.com") # (1)!

print(attachment.text)
```

1. **Attachments** depend on the `markitdown` library for various text based sources such as HTML, PDFs, Markdown, etc. for quick and accurate rendering to text that can be passed to a model.

### Multimodal Content

You can also pass in multimodal content such as images, audio, video, documents, etc. from a local file or URL!

```python title="Using paste() on multimodal content"
from zyx import paste

attachment = paste("https://picsum.photos/id/237/536/354")

print(attachment.text)
"""
[image content at https://picsum.photos/id/237/536/354]
"""
```

### Pythonic Objects

You can also pass in standard Pythonic objects to represent them in the `TOON` (Token-Oriented Object Notation) format that can be cleanly interpreted by a model.

```python title="Using paste() on a Pythonic object"
from zyx import paste

attachment = paste({"name": "John", "age": 30}) # (1)!

print(attachment.text)
"""
name: John
age: 30
"""
```

## Passing Attachments to Semantic Operations

An attachment can be included in the `context` of any semantic operation, or as the direct `source` for operations such as `parse` or `query` that support the parameter.

```python title="Passing an Attachment to a semantic operation" hl_lines="6"
from zyx import make, paste

attachment = paste("https://picsum.photos/id/237/536/354")

result = make(
    context=[attachment, "What is this image?"], # (1)!
)

print(result.output)
"""
The image shows a black puppy sitting on a wooden surface, looking up at the camera. Its large, expressive eyes and shiny coat suggest it's a young and playful dog.
"""
```

1. An attachment can be included just like any other context item!

## Interactive Attachments

If you want a model to interact with a source (query or edit), use `attach()`.

```python title="Attaching a local file"
from zyx import attach, make

notes = attach("notes.md")

result = make(
    attachments=[notes],
    context="Summarize the notes and fix obvious typos.",
)

print(result.output)
```

Interactive attachments expose tool interfaces when available, letting agents read and update the underlying source safely.
