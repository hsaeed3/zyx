__all__ = [
    "Client",
    "completion",
    "classify",
    "code",
    "extract",
    "function",
    "generate",
    "system_prompt",
]

from ....client import (
    Client as Client,
    completion as completion,
)

from ....resources.completions.base.classify import classify as classify
from ....resources.completions.base.code import code as code
from ....resources.completions.base.extract import extract as extract
from ....resources.completions.base.function import function as function
from ....resources.completions.base.generate import generate as generate
from ....resources.completions.base.system_prompt import system_prompt as system_prompt
