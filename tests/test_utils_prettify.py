"""tests.test_processing_prettify"""

import pytest
from rich.console import RenderableType
from rich import print

from zyx.utils.prettify import (
    prettify_result,
)
from zyx.results import Result


def test_prettify_result():
    result = Result(
        kind = "make",
        output = 45.6,
        models = ["gpt-4o-mini"]
    )

    assert isinstance(prettify_result(result), RenderableType)
    
    print("\n")
    print(prettify_result(result))

    
if __name__ == "__main__":
    pytest.main(
        ["--capture=no", __file__]
    )