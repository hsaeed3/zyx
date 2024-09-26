__all__ = [
    "image",
    "speech",
    "transcribe"
]


from .._loader import loader


class image(loader):
    pass


image.init("zyx.lib.multimodal", "image")


class speech(loader):
    pass


speech.init("zyx.lib.multimodal", "speech")


class transcribe(loader):
    pass


transcribe.init("zyx.lib.multimodal", "transcribe")