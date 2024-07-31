# zyx ==============================================================================

__all__ = [
    "Audio",
    "speak",
    "transcribe",
]

from ..core import _UtilLazyLoader 

class Audio(_UtilLazyLoader):
    pass
Audio.init("marvin.types", "Audio")

class speak(_UtilLazyLoader):
    pass
speak.init("marvin.ai.audio", "speak")

class transcribe(_UtilLazyLoader):
    pass
transcribe.init("marvin.ai.audio", "transcribe")