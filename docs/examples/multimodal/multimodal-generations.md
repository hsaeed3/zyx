# Multimodal Generations

`zyx` leverages simple functions for multimodal generations. These functions
have also been built in as LLM compatible tools as well, to provide multimodal tool calling agents.

---

## Image Generation

The image function is a direct routeof the `litellm.image_generation()` function.

```python
from zyx import image

image("An astronaut riding a rainbow unicorn")
```

---

## Audio Generation

Use the `audio()` function to generate audio. This is a direct text -> speech.

```python
from zyx import audio

audio("Hello, my name is john!")
```

**API Reference**

::: zyx.resources.ext.multimodal.audio

---

## Audio Trancription

Use the `transcribe()` function to convert audio files into text.

```python
from zyx import transcribe

transcribe("path/to/audio.mp3")
```

**API Reference**

::: zyx.resources.ext.multimodal.transcribe
