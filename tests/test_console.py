import pytest
from zyx.console import Console

@pytest.fixture
def console():
    return Console()

def test_message(console):
    # Capture the output of the console message
    with console.capture() as capture:
        console.message("Test message")
    output = capture.get()
    assert "Test message" in output

def test_warning(console):
    # Capture the output of the console warning
    with console.capture() as capture:
        console.warning("Test warning")
    output = capture.get()
    assert "Test warning" in output

# runnable test
if __name__ == "__main__":
    test_message(Console())
    test_warning(Console())