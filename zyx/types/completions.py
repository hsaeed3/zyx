from __future__ import annotations

"""
zyx.types.completions

This module provides chat completions types for `zyx`
"""

# [Imports]
from openai.types.chat.chat_completion import ChatCompletion
from typing import Any, Optional, Sequence, Mapping, Literal, List, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ConfigDict, model_serializer
from pathlib import Path
from base64 import b64encode, b64decode


# ===================================================================
# [Pydantic Subscriptable Extension]
# ===================================================================

class SubscriptableBaseModel(BaseModel):
    
    """
    Subscriptable BaseModel. Used as a base class for `zyx` completions types & models.
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


# ===================================================================
# [MULTIMODAL IMAGE INPUT]
# ===================================================================

class Image(BaseModel):
  value: Union[str, bytes, Path]

  @model_serializer
  def serialize_model(self):
    if isinstance(self.value, (Path, bytes)):
      return b64encode(self.value.read_bytes() if isinstance(self.value, Path) else self.value).decode()

    if isinstance(self.value, str):
      try:
        if Path(self.value).exists():
          return b64encode(Path(self.value).read_bytes()).decode()
      except Exception:
        # Long base64 string can't be wrapped in Path, so try to treat as base64 string
        pass

      # String might be a file path, but might not exist
      if self.value.split('.')[-1] in ('png', 'jpg', 'jpeg', 'webp'):
        raise ValueError(f'File {self.value} does not exist')

      try:
        # Try to decode to check if it's already base64
        b64decode(self.value)
        return self.value
      except Exception:
        raise ValueError('Invalid image data, expected base64 string or path to image file') from Exception


# ===================================================================
# [Messages]
# ===================================================================

# [Message Role]
MessageRole = Literal['user', 'assistant', 'system', 'tool']


# [Message]
class Message(SubscriptableBaseModel):
  
  """
  Chat message.
  """

  role: Literal['user', 'assistant', 'system', 'tool']
  "Assumed role of the message. Response messages has role 'assistant' or 'tool'."

  content: Optional[str] = None
  'Content of the message. Response messages contains message fragments when streaming.'

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
      'Name of the function.'

      arguments: Mapping[str, Any]
      'Arguments of the function.'

    function: Function
    'Function to be called.'

  tool_calls: Optional[Sequence[ToolCall]] = None
  """
  Tools calls to be made by the model.
  """
  
  
# ==============================================================
# [Chat Models]
# This might be removed in the future just because of how fast
# development is moving.
# ==============================================================

ChatModels = Literal[
    # Anthropic
    "anthropic/claude-3-5-haiku-latest",    # Vision
    "anthropic/claude-3-5-sonnet-latest",    # Vision
    "anthropic/claude-3-opus-latest",
    # Cohere
    "cohere/command-r-plus",
    "cohere/command-r",
    # Databricks
    "databricks/databricks-dbrx-instruct",
    # Gemini
    "gemini/gemini-pro",
    "gemini/gemini-1.5-pro-latest",
    # OpenAI
    "openai/gpt-4o-mini",    # Vision
    "openai/gpt-4o",    # Vision
    "openai/o1-mini",
    "openai/o1-preview",
    "openai/chatgpt-4o-latest",
    "openai/gpt-4-turbo",
    "openai/gpt-4",
    "openai/gpt-4-vision",    # Vision
    "openai/gpt-3.5-turbo",
    # Ollama
    "ollama/bespoke-minicheck",
    "ollama/llama3",
    "ollama/llama3.1",
    "ollama/llama3.2",
    "ollama/llama3.2-vision",    # Vision
    "ollama/llama-guard3",
    "ollama/llava",    # Vision
    "ollama/llava-llama3",    # Vision
    "ollama/llava-phi3",    # Vision
    "ollama/gemma2",
    "ollama/granite3-dense",
    "ollama/granite3-guardian",
    "ollama/granite3-moe",
    "ollama/minicpm-v",    # Vision
    "ollama/mistral",
    "ollama/mistral-nemo",
    "ollama/mistral-small",
    "ollama/mixtral",
    "ollama/moondream",    # Vision
    "ollama/nemotron",
    "ollama/nuextract",
    "ollama/opencoder",
    "ollama/phi3",
    "ollama/reader-lm",
    "ollama/smollm2",
    "ollama/shieldgemma",
    "ollama/tinyllama",
    "ollama/qwen",
    "ollama/qwen2"
    "ollama/qwen2.5",
    # Perplexity
    "perplexity/pplx-7b-chat",
    "perplexity/pplx-70b-chat",
    "perplexity/pplx-7b-online",
    "perplexity/pplx-70b-online",
    # XAI
    "xai/grok-beta",
]


# ==============================================================
# [CONTEXT (dsl)]
# ==============================================================

class Context(SubscriptableBaseModel):
    
    """
    Completion context.
    
    Handles initialization of context by attempting to parse various input types:
    - JSON strings are parsed using json.loads
    - Pydantic models are converted to JSON schema or dumped to JSON
    - Plain strings are used as-is
    """
    
    context: Optional[str] = None
    
    def __init__(self, **kwargs):
        import json
        from pydantic import BaseModel
        
        super().__init__(**kwargs)
        context = kwargs.get("context", None)
        
        if context is not None:
            if isinstance(context, str):
                try:
                    # Try parsing as JSON string first
                    json.loads(context)
                    self.context = context
                except json.JSONDecodeError:
                    # If not valid JSON, use as plain string
                    self.context = context
            elif isinstance(context, type) and issubclass(context, BaseModel):
                # Handle Pydantic model class
                self.context = json.dumps(context.model_json_schema())
            elif isinstance(context, BaseModel):
                # Handle Pydantic model instance
                self.context = context.model_dump_json()
            else:
                # Try converting other objects to JSON
                try:
                    self.context = json.dumps(context)
                except TypeError:
                    self.context = str(context)
                    
                    
# ==============================================================
# [Reasoning]
# ==============================================================

class Reasoning(BaseModel):
    
    """
    A well structured, sequenced & instructive list of steps providing 
    an incredibly intuitive process for achieving the task or desired output.
    """
    
    steps : List[str]
    
    
# ==============================================================
# [`Completion` Response Extension]
# ==============================================================

class ChatCompletions(ChatCompletion):
    
    """
    `ChatCompletion` extension generated by `zyx`.

    This class tries to build upon the 'ease of use' of the `Instructor` library when
    generating structured outputs, as well as adding some extended outputs that are `zyx`
    specific.
    """
    
    # == .pydantic ==
    # Initialized pydantic basemodel for structured outputs if `response_format` is provided
    # The hope was to use the `model` attribute, but that is ChatCompletion API spec standard for model name
    pydantic: Annotated[Optional[BaseModel], "Pydantic model for structured outputs"] = None

    # == .messages ==
    # List of messages generated or utilized in the completion
    # Useful for multi-message completions like `run_tools` & `reasoning`
    messages: Annotated[Optional[Sequence[Message]], "Messages generated or used in completion"] = None

    def __init__(self, **kwargs):
        """Initialize the CompletionResponse with any additional fields."""
        # Initialize parent class with standard fields
        super().__init__(**kwargs)
        
        # Allow any additional fields to be set
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)