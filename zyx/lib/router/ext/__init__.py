__all__ = ["BaseModel", "Field", "app", "image", "audio", "transcribe"]


from .._router import router

from ...types.base_model import BaseModel
from pydantic import Field


class app(router):
    pass


app.init("zyx.resources.ext.app", "terminal")


class image(router):
    pass


image.init("litellm.main", "image_generation")


class audio(router):
    pass


audio.init("zyx.resources.ext.multimodal", "audio")


class transcribe(router):
    pass


transcribe.init("zyx.resources.ext.multimodal", "transcribe")
