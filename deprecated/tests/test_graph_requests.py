"""tests.test_graph_requests"""

from zyx._graph._requests import (
    _render_output_context,
    _render_source_context,
    _render_source_metadata_values,
    _render_attachment_context,
)
from zyx._utils._outputs import OutputBuilder


def test_render_output_context_value_target():
    builder = OutputBuilder(target={"name": str})
    builder.update({"name": "A"})
    text = _render_output_context(builder)
    assert "Output Context" in text
    assert "starting state" in text
    assert "current state" in text


def test_render_source_context_markers():
    text = _render_source_context({"a": 1})
    assert "[PRIMARY INPUT]" in text
    assert "[END PRIMARY INPUT]" in text


def test_render_source_metadata_values():
    text = _render_source_metadata_values(
        origin="file", media_type="text/plain", source_repr="x"
    )
    assert "Origin: file" in text
    assert "Media Type: text/plain" in text


def test_render_attachment_context():
    text = _render_attachment_context(
        name="Attachment", description="desc", state="state"
    )
    assert "[ATTACHMENT: Attachment]" in text
    assert "Description:" in text
    assert "State:" in text
