"""
zyx.types.chat_completions.chat_message

Chat message type used for chat completions.
"""

from __future__ import annotations

# [Imports]
from pydantic import ConfigDict
from typing import Optional, Sequence, Literal, Mapping, Any
from ..subscriptable_base_model import SubscriptableBaseModel
from ..multimodal import Image


# [Chat Message Role]
ChatMessageRole = Literal["user", "assistant", "system", "tool"]


# [Chat Message]
class ChatMessage(SubscriptableBaseModel):
    """
    Chat message.
    """

    model_config = ConfigDict(exclude_none=True)

    role: Literal["user", "assistant", "system", "tool"]
    "Assumed role of the message. Response messages has role 'assistant' or 'tool'."

    content: Optional[str] = None
    "Content of the message. Response messages contains message fragments when streaming."

    images: Optional[Sequence[Image]] = None
    """
  Optional list of image data for multimodal models.

  Valid input types are:

  - `str` or path-like object: path to image file
  - `bytes` or bytes-like object: raw image data

  Valid image formats depend on the model. See the model card for more information.
  """

    class ToolCall(SubscriptableBaseModel):
        """
        Model tool calls.
        """

        class Function(SubscriptableBaseModel):
            """
            Tool call function.
            """

            name: str
            "Name of the function."

            arguments: Mapping[str, Any]
            "Arguments of the function."

        function: Function
        "Function to be called."

    tool_calls: Optional[Sequence[ToolCall]] = None
    """
  Tools calls to be made by the model.
  """

    tool_call_id: Optional[str] = None
    "ID of the tool call this message is responding to"

    def __init__(
        self,
        role: str,
        content: Optional[str] = None,
        images: Optional[Sequence[Image]] = None,
        tool_calls: Optional[Sequence[ToolCall]] = None,
        tool_call_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            role=role,
            **{
                k: v
                for k, v in dict(
                    content=content, images=images, tool_calls=tool_calls, tool_call_id=tool_call_id, **kwargs
                ).items()
                if v is not None
            },
        )
