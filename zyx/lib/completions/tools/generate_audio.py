def generate_audio(prompt: str) -> str:
    """A tool that generates an audio file from a text prompt,
    and returns the URL or path to the audio file."""

    from ...multimodal import speech

    return str(speech(
        prompt = prompt
    ))