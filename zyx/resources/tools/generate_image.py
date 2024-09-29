def generate_image(prompt: str) -> str:
    """A tool that generates an image from a prompt,
    and returns the URL or path to the image."""

    from ..ext.multimodal import image

    return str(image(prompt=prompt))
