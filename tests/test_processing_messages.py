"""tests.test_processing_messages"""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ImageUrl,
)

from zyx._processing._messages import (
    openai_dict_to_pydantic_ai_message,
    text_to_pydantic_ai_messages,
    compact_pydantic_ai_messages,
    parse_context_to_pydantic_ai_messages,
    parse_instructions_as_system_prompt_parts,
)


def test_openai_dict_to_message_system():
    msg = openai_dict_to_pydantic_ai_message(
        {"role": "system", "content": "hi"}
    )
    assert isinstance(msg, ModelRequest)
    assert isinstance(msg.parts[0], SystemPromptPart)
    assert msg.parts[0].content == "hi"


def test_openai_dict_to_message_user_with_image():
    msg = openai_dict_to_pydantic_ai_message(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "http://x"}},
            ],
        }
    )
    assert isinstance(msg, ModelRequest)
    content = msg.parts[0].content
    assert isinstance(content, list)
    assert "hello" in content
    assert any(isinstance(item, ImageUrl) for item in content)


def test_openai_dict_to_message_assistant_tool_call():
    msg = openai_dict_to_pydantic_ai_message(
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "tool", "arguments": "{}"},
                }
            ],
        }
    )
    assert isinstance(msg, ModelResponse)
    assert any(isinstance(part, ToolCallPart) for part in msg.parts)
    assert any(isinstance(part, TextPart) for part in msg.parts)


def test_text_to_messages_tags():
    text = "[s]sys[/s][u]hi[/u][a]ok[/a]"
    messages = text_to_pydantic_ai_messages(text)
    assert len(messages) == 3
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[1], ModelRequest)
    assert isinstance(messages[2], ModelResponse)


def test_text_to_messages_untagged():
    messages = text_to_pydantic_ai_messages("hello")
    assert len(messages) == 1
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[0].parts[0], UserPromptPart)


def test_compact_messages():
    messages = [
        ModelRequest(parts=[UserPromptPart(content="a")]),
        ModelRequest(parts=[UserPromptPart(content="b")]),
    ]
    compacted = compact_pydantic_ai_messages(messages)
    assert len(compacted) == 1
    assert isinstance(compacted[0], ModelRequest)


def test_parse_context_to_messages_openai_dict():
    context = [{"role": "user", "content": "hi"}]
    messages = parse_context_to_pydantic_ai_messages(context)
    assert len(messages) == 1
    assert isinstance(messages[0], ModelRequest)


def test_parse_instructions_as_system_prompt_parts_callable():
    def instr(ctx):
        return f"x={ctx.deps['x']}" if ctx is not None else "x=?"

    parts = parse_instructions_as_system_prompt_parts(
        instructions=instr, deps={"x": 1}
    )
    assert len(parts) == 1
    assert parts[0].content == "x=1"

    compact = parse_instructions_as_system_prompt_parts(
        instructions=["a", "b"], deps=None, compact=True
    )
    assert len(compact) == 1
    assert compact[0].content == "a\n\nb"
