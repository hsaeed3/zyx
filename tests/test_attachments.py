"""tests.test_attachments"""

from pydantic_ai.toolsets import FunctionToolset

from zyx.attachments import attach, paste


def test_paste_attachment_string_properties():
    attachment = paste("hello")
    assert attachment.message is not None
    assert "string value" in attachment.description
    assert attachment.text == "hello"
    assert attachment.get_toolset() is None


def test_attach_object_toolset():
    attachment = attach({"a": 1})
    toolset = attachment.get_toolset()
    assert isinstance(toolset, FunctionToolset)
