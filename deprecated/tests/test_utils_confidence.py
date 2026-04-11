"""tests.test_utils_confidence"""

from pydantic_ai.messages import ModelResponse, TextPart

from zyx._utils import _confidence as conf


class DummyRun:
    def __init__(self, messages):
        self._messages = messages

    def all_messages(self):
        return self._messages


def _make_run_with_logprobs(tokens):
    msg = ModelResponse(
        parts=[TextPart(content="".join(t["token"] for t in tokens))],
        provider_details={"logprobs": tokens},
    )
    return DummyRun([msg])


def test_score_confidence_level_thresholds():
    assert conf._score_confidence_level(0.95) is conf.ConfidenceLevel.HIGH
    assert conf._score_confidence_level(0.8) is conf.ConfidenceLevel.MEDIUM
    assert conf._score_confidence_level(0.6) is conf.ConfidenceLevel.LOW
    assert conf._score_confidence_level(0.2) is conf.ConfidenceLevel.VERY_LOW


def test_score_confidence_primitive():
    tokens = [{"token": "hello", "logprob": -0.1}]
    run = _make_run_with_logprobs(tokens)
    confidence = conf.score_confidence([run], output="hello")
    assert confidence is not None
    assert confidence.level is conf.ConfidenceLevel.HIGH
    assert confidence.token_count == 1


def test_score_confidence_structured_fields():
    tokens = [
        {"token": "name", "logprob": -0.2},
        {"token": " Alice", "logprob": -0.2},
        {"token": " age", "logprob": -0.2},
        {"token": " 30", "logprob": -0.2},
    ]
    run = _make_run_with_logprobs(tokens)
    output = {"name": "Alice", "age": 30}
    confidence = conf.score_confidence([run], output=output)
    assert confidence is not None
    assert set(confidence.fields.keys()) == {"name", "age"}
