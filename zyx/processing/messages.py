"""zyx.processing.messages"""

from __future__ import annotations

import re
from typing import (
    Any,
    Dict,
    List,
    Pattern,
)

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import (
    messages as _pydantic_ai_messages,
)


PydanticAIMessageAdapter : TypeAdapter[
    _pydantic_ai_messages.ModelMessage
] = TypeAdapter(_pydantic_ai_messages.ModelMessage)


_ROLE_TAG_PATTERN : Pattern[str] = re.compile(
    pattern = r'\[(s|system|u|user|a|assistant)\](.*?)\[/\1\]',
    flags = re.DOTALL | re.IGNORECASE
)
_UNCLOSED_TAG_PATTERN : Pattern[str] = re.compile(
    pattern = r'\[(s|system|u|user|a|assistant)\](.*)$',
    flags = re.DOTALL | re.IGNORECASE
)
_ROLE_MAP : Dict[str, str] = {
    "s" : "system",
    "system" : "system",
    "u" : "user",
    "user" : "user",
    "a" : "assistant",
    "assistant" : "assistant",
}


def parse_openai_like_to_message(message : BaseModel | Dict[str, Any] | Any) -> _pydantic_ai_messages.ModelMessage:
    """
    Parses an OpenAI Chat Completions like message dictionary or object into a PydanticAI ModelMessage object.
    """
    # assume maybe a message could appear as a pydantic model, this might never
    # happen, but also truly we never know
    if hasattr(message, "model_dump"):
        message = message.model_dump() # type: ignore

    if not "role" in message or not "content" in message:
        raise ValueError(
            (
                "Recieved invalid dictionary representation of a message. An OpenAI-like message "
                "must contain both a `role` and `content` key to be considered valid."
            )
        )

    role : str = message['role'].lower()
    content : str | List[Dict[str, Any]] = message['content']
    
    if role == "system":
        if isinstance(content, str):
            return _pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.SystemPromptPart(content=content)])
        elif isinstance(content, list):
            parts : List[_pydantic_ai_messages.SystemPromptPart] = []
            for part in content:
                if not "type" in part:
                    raise ValueError(
                        "Found invalid content part within system message content. ",
                        "Each part must contain a `type` key to be considered valid."
                    )
                if part.get("type") == "text":
                    parts.append(_pydantic_ai_messages.SystemPromptPart(content=part["text"]))
            
            return _pydantic_ai_messages.ModelRequest(parts=parts)

    elif role == "user":
        if isinstance(content, str):
            return _pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.UserPromptPart(content=content)])

        elif isinstance(content, list):
            parts : List[_pydantic_ai_messages.UserContent] = []
            for part in content:
                if not "type" in part:
                    raise ValueError(
                        "Found invalid content part within user message content. ",
                        "Each part must contain a `type` key to be considered valid."
                    )
            
                if part["type"] == "text":
                    parts.append(part["text"])
                
                elif part["type"] == "image_url":
                    parts.append(_pydantic_ai_messages.ImageUrl(
                        url=part["image_url"]["url"],
                    ))
                
                elif part["type"] == "input_audio":
                    parts.append(_pydantic_ai_messages.BinaryContent(
                        data=part["input_audio"]["data"],
                        media_type=part["input_audio"]["format"]
                    ))
                
                elif part["type"] == "file":
                    parts.append(_pydantic_ai_messages.BinaryContent(
                        data=part["file"]["file_data"],
                        media_type=part["file"]["file_id"]
                    ))

            return _pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.UserPromptPart(content=parts)])

    elif role == "assistant":
        parsed_content : List[_pydantic_ai_messages.ModelResponsePart] = []

        if "tool_calls" in message and isinstance(message["tool_calls"], list):
            # assume tool calls are in correct format, if theyre being passed in
            # as a dict. no need to validate every little bit
            for tool_call in message["tool_calls"]:
                parsed_content.append(
                    _pydantic_ai_messages.ToolCallPart(
                        tool_name=tool_call["function"]["name"],
                        args=tool_call["function"]["arguments"],
                        tool_call_id=tool_call["id"]
                    )
                )

        if content is not None:
            if isinstance(content, str):
                parsed_content.append(_pydantic_ai_messages.TextPart(content=content))

            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parsed_content.append(_pydantic_ai_messages.TextPart(content=part["text"]))

                # TODO: decide if theres a need to support additional content parts
                # at this level of dictionary parsing

        return _pydantic_ai_messages.ModelResponse(parts=parsed_content)

    elif role == "tool":
        if isinstance(content, str):
            return _pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.ToolReturnPart(
                tool_name=message["name"],
                content=content
            )])
            
        elif isinstance(content, list):
            parts : List[_pydantic_ai_messages.ToolReturnContent] = []
            for part in content:
                if part.get("type") == "text":
                    parts.append(part["text"])

            return _pydantic_ai_messages.ModelRequest(parts=[
                _pydantic_ai_messages.ToolReturnPart(
                    tool_name=message["name"],
                    content=parts
                )
            ])

    raise ValueError(
        "Recieved invalid dictionary representation of an OpenAI ChatCompletions message. ",
        f"A message must match one of the following roles: 'user', 'system', 'assistant', 'tool', received: '{role}'"
    )


def parse_string_to_messages(text : str) -> List[_pydantic_ai_messages.ModelMessage]:
    """
    Parses a string template into a list of ModelRequest/ModelResponse objects,
    accounting for any 'role tags' present in the text.
    """
    messages : List[_pydantic_ai_messages.ModelMessage] = []
    # index of the last end tagged section
    last_end : int = 0

    for match in _ROLE_TAG_PATTERN.finditer(text):
        # if any text is untagged, and matches are present, this is considered
        # a user message
        if match.start() > last_end:
            untagged = text[last_end:match.start()].strip()
            if untagged:
                messages.append(_pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.UserPromptPart(untagged)]))
        
        tag = match.group(1).lower()
        content = match.group(2).strip()
        role = _ROLE_MAP[tag]

        if content:
            if role == "system":
                messages.append(_pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.SystemPromptPart(content=content)]))
            elif role == "user":
                messages.append(_pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.UserPromptPart(content=content)]))
            elif role == "assistant":
                messages.append(_pydantic_ai_messages.ModelResponse(parts=[_pydantic_ai_messages.TextPart(content=content)]))

            # NOTE:
            # any other tags that arent system, user, or assistant are ignored, and
            # the text is considered untagged
            else:
                messages.append(_pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.UserPromptPart(content=content)]))

        last_end = match.end()

    if last_end < len(text):
        # if additional content/text is present, beyond any tags, check if it's
        # an unclosed tag at the end
        remaining = text[last_end:].strip()
        if remaining:
            # check if remaining text starts with an unclosed tag
            unclosed_match = _UNCLOSED_TAG_PATTERN.match(remaining)
            if unclosed_match:
                tag = unclosed_match.group(1).lower()
                content = unclosed_match.group(2).strip()
                role = _ROLE_MAP[tag]
                
                if content:
                    if role == "system":
                        messages.append(_pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.SystemPromptPart(content=content)]))
                    elif role == "user":
                        messages.append(_pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.UserPromptPart(content=content)]))
                    elif role == "assistant":
                        messages.append(_pydantic_ai_messages.ModelResponse(parts=[_pydantic_ai_messages.TextPart(content=content)]))
                    else:
                        # fallback to user message for unknown roles
                        messages.append(_pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.UserPromptPart(content=content)]))
            else:
                # no unclosed tag, treat as user message
                messages.append(_pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.UserPromptPart(remaining)]))
    
    # finally, once again we fallback to a user message
    if not messages and text.strip():
        messages.append(_pydantic_ai_messages.ModelRequest(parts=[_pydantic_ai_messages.UserPromptPart(text.strip())]))

    return messages  


def compact_messages(messages : List[_pydantic_ai_messages.ModelMessage]) -> List[_pydantic_ai_messages.ModelMessage]:
    """
    Compacts consecutive ModelRequest or ModelResponse objects into single
    objects, if applicable.
    """
    if not messages:
        return messages

    compacted : List[_pydantic_ai_messages.ModelMessage] = []

    for message in messages:
        if not compacted:
            compacted.append(message)
            continue

        last_message = compacted[-1]

        if isinstance(message, _pydantic_ai_messages.ModelRequest) and isinstance(last_message, _pydantic_ai_messages.ModelRequest):
            compacted[-1] = _pydantic_ai_messages.ModelRequest(parts=list(last_message.parts) + list(message.parts))
        elif isinstance(message, _pydantic_ai_messages.ModelResponse) and isinstance(last_message, _pydantic_ai_messages.ModelResponse):
            compacted[-1] = _pydantic_ai_messages.ModelResponse(parts=list(last_message.parts) + list(message.parts))
        else:
            compacted.append(message)

    return compacted


def build_system_prompt(
    instructions : Any | List[Any] | None = None,
    prepend_instructions : str | List[str] | None = None,
    append_instructions : str | List[str] | None = None,
    constraints : str | None = None,
) -> _pydantic_ai_messages.SystemPromptPart | None:
    """
    Prepares a system prompt for a PydanticAI agent's message history.
    """
    system_prompt : str = ""

    if instructions:
        if isinstance(instructions, (str, List[str])):
            system_prompt += f"\n\n{instructions}"

    if prepend_instructions:
        if isinstance(prepend_instructions, str):
            system_prompt += f"{prepend_instructions}\n"
        elif isinstance(prepend_instructions, List[str]):
            system_prompt += f"{' '.join(prepend_instructions)}\n"

    if append_instructions:
        if isinstance(append_instructions, str):
            system_prompt += f"{append_instructions}\n"
        elif isinstance(append_instructions, List[str]):
            system_prompt += f"{' '.join(append_instructions)}\n"

    if constraints:
        # this assumes properly rendered string from source
        system_prompt += f"\n\n{constraints}"

    if system_prompt:
        return _pydantic_ai_messages.SystemPromptPart(content=system_prompt)
    return None    