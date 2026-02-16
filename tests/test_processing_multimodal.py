"""tests.test_processing_multimodal"""

from zyx._processing._multimodal import (
    MultimodalContentOrigin,
    MultimodalContentMediaType,
    classify_multimodal_source,
    render_multimodal_source_as_text,
    render_multimodal_source_as_description,
)


def test_classify_string_text():
    origin, media_type = classify_multimodal_source("hello")
    assert origin is MultimodalContentOrigin.STRING
    assert media_type is MultimodalContentMediaType.TEXT


def test_classify_bytes_pdf():
    data = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3"
    origin, media_type = classify_multimodal_source(data)
    assert origin is MultimodalContentOrigin.BYTES
    assert media_type is MultimodalContentMediaType.DOCUMENT


def test_render_multimodal_source_as_text_bytes():
    data = b"hello"
    text = render_multimodal_source_as_text(
        data,
        MultimodalContentOrigin.BYTES,
        MultimodalContentMediaType.TEXT,
    )
    assert text == "hello"


def test_render_multimodal_source_as_description_bytes():
    data = b"abc"
    description = render_multimodal_source_as_description(
        data,
        MultimodalContentOrigin.BYTES,
        MultimodalContentMediaType.TEXT,
    )
    assert "size=3" in description
