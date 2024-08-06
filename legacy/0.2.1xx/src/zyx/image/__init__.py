# zyx ==============================================================================

__all__ = [
    "caption",
    "paint",
    "Image",
    "ImageUrl",
]

from ..core import _UtilLazyLoader

class caption(_UtilLazyLoader):
    pass
caption.init("marvin.ai.text", "caption")

class paint(_UtilLazyLoader):
    pass
paint.init("marvin.ai.images", "paint")

class Image(_UtilLazyLoader):
    pass
Image.init("marvin.types", "Image")

class ImageUrl(_UtilLazyLoader):
    pass
ImageUrl.init("marvin.types", "ImageUrl")
