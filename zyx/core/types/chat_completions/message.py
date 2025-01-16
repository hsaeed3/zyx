"""
### zyx.core.types.chat_completions.message

Message type
"""

from __future__ import annotations

# [Imports]
from typing import Any
from pydantic import BaseModel, ConfigDict
from typing import Optional, Sequence, Literal, Mapping, Any
from ..multimodal import Image


# ===================================================================
# [Subscriptable BaseModel]
# ===================================================================


class SubscriptableBaseModel(BaseModel):
    """
    Subscriptable BaseModel. Used as an internal base class for `zyx` types & models.
    """

    def __getitem__(self, key: str) -> Any:
        if key in self:
            return getattr(self, key)

        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        if key in self.model_fields_set:
            return True

        if key in self.model_fields:
            return self.model_fields[key].default is not None

        return False

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default


# -------------------------------------------------------------------
# Messages
# -------------------------------------------------------------------


# [Chat Message Role]
ChatMessageRole = Literal["user", "assistant", "system", "tool"]


# [Chat Message]
class Message(SubscriptableBaseModel):
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
