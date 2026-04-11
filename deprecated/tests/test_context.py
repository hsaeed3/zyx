"""tests.test_context"""

from pydantic_ai.messages import ModelRequest, UserPromptPart, SystemPromptPart
from pydantic_ai.toolsets import FunctionToolset

from zyx.context import Context


def test_context_render_messages_max_length():
    ctx = Context(messages=["hi", "there"], max_length=1)
    messages = ctx.render_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[0].parts[0], UserPromptPart)


def test_context_add_messages():
    ctx = Context()
    ctx.add_user_message("hello")
    ctx.add_assistant_message("ok")
    ctx.add_system_message("sys")
    assert len(ctx.render_messages()) == 3


def test_context_render_instructions_compact():
    ctx = Context(instructions="a\n\nb", compact_instructions=True)
    parts = ctx.render_instructions()
    assert len(parts) == 1
    assert isinstance(parts[0], SystemPromptPart)
    assert parts[0].content == "a\n\nb"


def test_context_render_toolsets_function():
    def tool_fn():
        return "ok"

    ctx = Context(tools=[tool_fn])
    toolsets = ctx.render_toolsets()
    assert len(toolsets) == 1
    assert isinstance(toolsets[0], FunctionToolset)
