"""
zyx.base.ext

Pydantic basemodel subscriptable extension used for configs & type models.
"""

from __future__ import annotations

# [Imports]
from typing import Any
from pydantic import BaseModel


# ===================================================================
# [Subscriptable BaseModel]
# ===================================================================


class SubscriptableBaseModel(BaseModel):
    """
    Subscriptable BaseModel. Used as an internal base class for `zyx` types & models.
    """

    def __getitem__(self, key: str) -> Any:
        """
        >>> msg = Message(role='user')
        >>> msg['role']
        'user'
        >>> msg = Message(role='user')
        >>> msg['nonexistent']
        Traceback (most recent call last):
        KeyError: 'nonexistent'
        """
        if key in self:
            return getattr(self, key)

        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        >>> msg = Message(role='user')
        >>> msg['role'] = 'assistant'
        >>> msg['role']
        'assistant'
        >>> tool_call = Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))
        >>> msg = Message(role='user', content='hello')
        >>> msg['tool_calls'] = [tool_call]
        >>> msg['tool_calls'][0]['function']['name']
        'foo'
        """
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """
        >>> msg = Message(role='user')
        >>> 'nonexistent' in msg
        False
        >>> 'role' in msg
        True
        >>> 'content' in msg
        False
        >>> msg.content = 'hello!'
        >>> 'content' in msg
        True
        >>> msg = Message(role='user', content='hello!')
        >>> 'content' in msg
        True
        >>> 'tool_calls' in msg
        False
        >>> msg['tool_calls'] = []
        >>> 'tool_calls' in msg
        True
        >>> msg['tool_calls'] = [Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))]
        >>> 'tool_calls' in msg
        True
        >>> msg['tool_calls'] = None
        >>> 'tool_calls' in msg
        True
        >>> tool = Tool()
        >>> 'type' in tool
        True
        """
        if key in self.model_fields_set:
            return True

        if key in self.model_fields:
            return self.model_fields[key].default is not None

        return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        >>> msg = Message(role='user')
        >>> msg.get('role')
        'user'
        >>> msg = Message(role='user')
        >>> msg.get('nonexistent')
        >>> msg = Message(role='user')
        >>> msg.get('nonexistent', 'default')
        'default'
        >>> msg = Message(role='user', tool_calls=[ Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))])
        >>> msg.get('tool_calls')[0]['function']['name']
        'foo'
        """
        return self[key] if key in self else default
