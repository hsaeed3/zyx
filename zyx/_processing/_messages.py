"""zyx.processing.messages"""

from __future__ import annotations

import re
from typing import (
    Any,
    Dict,
    List,
    Pattern,
    TYPE_CHECKING,
)

from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    ToolReturnContent,
    TextPart,
    UserContent,
    UserPromptPart,
    BinaryContent,
    ImageUrl,
)


PydanticAIMessageAdapter: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)


_STRING_MESSAGE_ROLE_PATTERN: Pattern[str] = re.compile(
    pattern=r"\[(s|system|u|user|a|assistant)\](.*?)\[/\1\]",
    flags=re.DOTALL | re.IGNORECASE,
)


_STRING_MESSAGE_UNCLOSED_ROLE_PATTERN: Pattern[str] = re.compile(
    pattern=r"\[(s|system|u|user|a|assistant)\](.*)$",
    flags=re.DOTALL | re.IGNORECASE,
)


_STRING_MESSAGE_ROLE_MAP: Dict[str, str] = {
    "s": "system",
    "system": "system",
    "u": "user",
    "user": "user",
    "a": "assistant",
    "assistant": "assistant",
}


def openai_dict_to_pydantic_ai_message(
    message: Dict[str, Any],
) -> ModelMessage:
    """
    Converts an OpenAI-like message dictionary to a PydanticAI ModelMessage object.
    """
    if "role" not in message or "content" not in message:
        raise ValueError(
            (
                "Recieved invalid dictionary representation of a message. An OpenAI-like message "
                "must contain both a `role` and `content` key to be considered valid."
            )
        )

    role: str = message["role"].lower()
    content: str | List[Dict[str, Any]] = message["content"]

    if role == "system":
        if isinstance(content, str):
            return ModelRequest(parts=[SystemPromptPart(content=content)])
        elif isinstance(content, list):
            parts: List[SystemPromptPart] = []
            for part in content:
                if "type" not in part:
                    raise ValueError(
                        "Found invalid content part within system message content. ",
                        "Each part must contain a `type` key to be considered valid.",
                    )
                if part.get("type") == "text":
                    parts.append(SystemPromptPart(content=part["text"]))

            return ModelRequest(parts=parts)

    elif role == "user":
        if isinstance(content, str):
            return ModelRequest(parts=[UserPromptPart(content=content)])

        elif isinstance(content, list):
            parts: List[UserContent] = []
            for part in content:
                if "type" not in part:
                    raise ValueError(
                        "Found invalid content part within user message content. ",
                        "Each part must contain a `type` key to be considered valid.",
                    )

                if part["type"] == "text":
                    parts.append(part["text"])

                elif part["type"] == "image_url":
                    parts.append(
                        ImageUrl(
                            url=part["image_url"]["url"],
                        )
                    )

                elif part["type"] == "input_audio":
                    parts.append(
                        BinaryContent(
                            data=part["input_audio"]["data"],
                            media_type=part["input_audio"]["format"],
                        )
                    )

                elif part["type"] == "file":
                    parts.append(
                        BinaryContent(
                            data=part["file"]["file_data"],
                            media_type=part["file"]["file_id"],
                        )
                    )

            return ModelRequest(parts=[UserPromptPart(content=parts)])

    elif role == "assistant":
        parsed_content: List[ModelResponsePart] = []

        if "tool_calls" in message and isinstance(message["tool_calls"], list):
            # assume tool calls are in correct format, if theyre being passed in
            # as a dict. no need to validate every little bit
            for tool_call in message["tool_calls"]:
                parsed_content.append(
                    ToolCallPart(
                        tool_name=tool_call["function"]["name"],
                        args=tool_call["function"]["arguments"],
                        tool_call_id=tool_call["id"],
                    )
                )

        if content is not None:
            if isinstance(content, str):
                parsed_content.append(TextPart(content=content))

            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parsed_content.append(TextPart(content=part["text"]))

                # TODO: decide if theres a need to support additional content parts
                # at this level of dictionary parsing

        return ModelResponse(parts=parsed_content)

    elif role == "tool":
        if isinstance(content, str):
            return ModelRequest(
                parts=[
                    ToolReturnPart(tool_name=message["name"], content=content)
                ]
            )

        elif isinstance(content, list):
            parts: List[ToolReturnContent] = []
            for part in content:
                if part.get("type") == "text":
                    parts.append(part["text"])

            return ModelRequest(
                parts=[
                    ToolReturnPart(tool_name=message["name"], content=parts)
                ]
            )

    raise ValueError(
        "Recieved invalid dictionary representation of an OpenAI ChatCompletions message. ",
        f"A message must match one of the following roles: 'user', 'system', 'assistant', 'tool', received: '{role}'",
    )


def text_to_pydantic_ai_messages(text: str) -> List[ModelMessage]:
    """
    Converts a string to a list of PydanticAI ModelMessage objects.

    This supports the use of 'role tags' to include multiple messages within
    a single string. Supported tags are:

    - [s] or [system] for system messages
    - [u] or [user] for user messages
    - [a] or [assistant] for assistant messages

    Ex : [s] You are a helpful assistant.[/s] [u] Hello, how are you?[/u]
    """
    messages: List[ModelMessage] = []
    # index of the last end tagged section
    last_end: int = 0

    for match in _STRING_MESSAGE_ROLE_PATTERN.finditer(text):
        # if any text is untagged, and matches are present, this is considered
        # a user message
        if match.start() > last_end:
            untagged = text[last_end : match.start()].strip()
            if untagged:
                messages.append(ModelRequest(parts=[UserPromptPart(untagged)]))

        tag = match.group(1).lower()
        content = match.group(2).strip()
        role = _STRING_MESSAGE_ROLE_MAP[tag]

        if content:
            if role == "system":
                messages.append(
                    ModelRequest(parts=[SystemPromptPart(content=content)])
                )
            elif role == "user":
                messages.append(
                    ModelRequest(parts=[UserPromptPart(content=content)])
                )
            elif role == "assistant":
                messages.append(
                    ModelResponse(parts=[TextPart(content=content)])
                )

            # NOTE:
            # any other tags that arent system, user, or assistant are ignored, and
            # the text is considered untagged
            else:
                messages.append(
                    ModelRequest(parts=[UserPromptPart(content=content)])
                )

        last_end = match.end()

    if last_end < len(text):
        # if additional content/text is present, beyond any tags, check if it's
        # an unclosed tag at the end
        remaining = text[last_end:].strip()
        if remaining:
            # check if remaining text starts with an unclosed tag
            unclosed_match = _STRING_MESSAGE_UNCLOSED_ROLE_PATTERN.match(
                remaining
            )
            if unclosed_match:
                tag = unclosed_match.group(1).lower()
                content = unclosed_match.group(2).strip()
                role = _STRING_MESSAGE_ROLE_MAP[tag]

                if content:
                    if role == "system":
                        messages.append(
                            ModelRequest(
                                parts=[SystemPromptPart(content=content)]
                            )
                        )
                    elif role == "user":
                        messages.append(
                            ModelRequest(
                                parts=[UserPromptPart(content=content)]
                            )
                        )
                    elif role == "assistant":
                        messages.append(
                            ModelResponse(parts=[TextPart(content=content)])
                        )
                    else:
                        # fallback to user message for unknown roles
                        messages.append(
                            ModelRequest(
                                parts=[UserPromptPart(content=content)]
                            )
                        )
            else:
                # no unclosed tag, treat as user message
                messages.append(
                    ModelRequest(parts=[UserPromptPart(remaining)])
                )

    # finally, once again we fallback to a user message
    if not messages and text.strip():
        messages.append(ModelRequest(parts=[UserPromptPart(text.strip())]))

    return messages


def compact_pydantic_ai_messages(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """
    Compacts consecutive ModelRequest or ModelResponse messages into single messages, if applicable.
    """
    if not messages:
        return messages

    compacted: List[ModelMessage] = []

    for message in messages:
        if not compacted:
            compacted.append(message)
            continue

        last_message = compacted[-1]

        if isinstance(message, ModelRequest) and isinstance(
            last_message, ModelRequest
        ):
            compacted[-1] = ModelRequest(
                parts=list(last_message.parts) + list(message.parts)
            )
        elif isinstance(message, ModelResponse) and isinstance(
            last_message, ModelResponse
        ):
            compacted[-1] = ModelResponse(
                parts=list(last_message.parts) + list(message.parts)
            )
        else:
            compacted.append(message)

    return compacted


def parse_context_to_pydantic_ai_messages(
    context: Any | List[Any] | None = None,
) -> List[ModelMessage]:
    """
    Prepares the `context` parameter of a semantic operation into a list of
    `ModelMessage` objects.
    """
    messages: List[ModelMessage] = []

    from ..snippets import Snippet
    from ..context import Context

    if not context:
        return []
    else:
        if not isinstance(context, list):
            context = [context]

        seen_snippet_ids: set[int] = set()

        for item in context:
            if isinstance(item, Context):
                messages.extend(item.render_messages())
                continue

            if isinstance(item, str):
                messages.extend(text_to_pydantic_ai_messages(item))
                continue

            if isinstance(item, Snippet):
                sid = id(item)
                if sid in seen_snippet_ids:
                    continue
                seen_snippet_ids.add(sid)
                messages.append(item.message)
                continue

            if hasattr(item, "model_dump"):
                item = item.model_dump()

            if isinstance(item, dict):
                if "parts" in item:
                    try:
                        messages.append(
                            PydanticAIMessageAdapter.validate_python(item)
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Failed to convert PydanticAI-like message dictionary to a ModelMessage object: {e}"
                        )
                elif "role" in item:
                    try:
                        messages.append(
                            openai_dict_to_pydantic_ai_message(item)
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Failed to convert OpenAI-like message dictionary to a ModelMessage object: {e}"
                        )
                else:
                    raise ValueError(f"Invalid message item: {item}")
        return compact_pydantic_ai_messages(messages)
