"""zyx.models.language.model"""

from __future__ import annotations

from typing import Type, TypeVar, Generic, AsyncIterator, Iterator, overload, Iterable
from collections.abc import Iterable

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from ...core._logging import _get_logger
from ..adapters import ModelAdapter
from ..adapters.openai import OpenAIModelAdapter
from ..adapters.litellm import LiteLLMModelAdapter
from ..providers import (
    ModelProvider,
    ModelProviderName,
    infer_language_model_provider,
)
from .types import (
    LanguageModelSettings,
    LanguageModelResponse,
    LanguageModelName,
)


_logger = _get_logger(__name__)


T = TypeVar("T")
"""Generic type variable for structured output types."""


class LanguageModel(Generic[T]):
    """A language model that can generate both standard text completions
    and structured outputs.
    
    The LanguageModel provides a unified interface for interacting with
    various LLM providers through different adapters (OpenAI, LiteLLM).
    
    Key Features:
        - Automatic provider inference from model names
        - Support for both OpenAI-compatible and LiteLLM providers
        - Structured output generation via instructor
        - Streaming support for both text and structured outputs
        - Flexible configuration through settings
    
    Examples:
        # Standard text completion
        model = LanguageModel("openai/gpt-4")
        response = await model.arun("What is 2+2?")
        print(response.output)  # "4"
        
        # Structured output
        from pydantic import BaseModel
        
        class Answer(BaseModel):
            result: int
            explanation: str
        
        model = LanguageModel("openai/gpt-4")
        response = await model.arun(
            "What is 2+2?",
            type=Answer
        )
        print(response.output.result)  # 4
        print(response.output.explanation)  # "The sum of 2 and 2 is 4"
        
        # Streaming
        model = LanguageModel("openai/gpt-4")
        async for chunk in model.arun("Write a story", stream=True):
            print(chunk.output, end="", flush=True)
    """
    
    _model: str
    _adapter: ModelAdapter
    _settings: LanguageModelSettings | None
    
    @overload
    def __init__(
        self,
        model: LanguageModelName | str,
        *,
        settings: LanguageModelSettings | None = None,
    ) -> None: ...
    
    @overload
    def __init__(
        self,
        model: LanguageModelName | str,
        *,
        adapter: ModelAdapter,
        settings: LanguageModelSettings | None = None,
    ) -> None: ...
    
    @overload
    def __init__(
        self,
        model: LanguageModelName | str,
        *,
        provider: ModelProvider | ModelProviderName | str,
        api_key: str | None = None,
        adapter_type: str = "auto",
        settings: LanguageModelSettings | None = None,
    ) -> None: ...
    
    def __init__(
        self,
        model: LanguageModelName | str,
        *,
        adapter: ModelAdapter | None = None,
        provider: ModelProvider | ModelProviderName | str | None = None,
        api_key: str | None = None,
        adapter_type: str = "auto",
        settings: LanguageModelSettings | None = None,
    ):
        """Initialize a LanguageModel instance.
        
        Args:
            model: Model name (e.g., "openai/gpt-4", "anthropic/claude-3-opus", "gpt-4")
            adapter: Pre-configured ModelAdapter instance. If provided, provider,
                api_key, and adapter_type are ignored.
            provider: Provider configuration (ModelProvider instance, provider name,
                or custom base URL). If not provided, will be inferred from model name.
            api_key: Optional API key override
            adapter_type: Which adapter to use ("auto", "openai", "litellm").
                - "auto": Automatically selects OpenAI adapter for compatible providers,
                  LiteLLM for others
                - "openai": Force OpenAI adapter (requires OpenAI-compatible provider)
                - "litellm": Force LiteLLM adapter (works with all providers)
            settings: Default settings for model invocations
        
        Raises:
            ValueError: If model format is invalid or provider cannot be inferred
        """
        self._model = model
        self._settings = settings
        self._adapter = None
        
        if adapter is not None:
            # Use provided adapter
            self._adapter = adapter
            _logger.debug(f"Initialized LanguageModel with provided adapter: {adapter.name}")
        else:
            # Infer or create adapter
            self._adapter = self._create_adapter(
                model=model,
                provider=provider,
                api_key=api_key,
                adapter_type=adapter_type,
            )
    
    def _create_adapter(
        self,
        model: str,
        provider: ModelProvider | ModelProviderName | str | None,
        api_key: str | None,
        adapter_type: str,
    ) -> ModelAdapter:
        """Create the appropriate adapter for this model.
        
        Args:
            model: Model name
            provider: Provider configuration or None to infer
            api_key: Optional API key
            adapter_type: Type of adapter to create
        
        Returns:
            Configured ModelAdapter instance
        """
        # Parse model string to extract provider if prefixed
        model_provider = None
        model_name = model
        
        if "/" in model:
            provider_prefix, model_name = model.split("/", 1)
            # If no explicit provider given, use the prefix
            if provider is None:
                provider = provider_prefix
                _logger.debug(f"Extracted provider '{provider}' from model string")
        
        # Infer provider if still not set
        if provider is None:
            inferred = infer_language_model_provider(model_name)
            if inferred is None:
                raise ValueError(
                    f"Cannot infer provider from model name '{model}'. "
                    f"Please specify provider explicitly or use format 'provider/model'."
                )
            provider = inferred
            _logger.debug(f"Inferred provider: {provider.name}")
        
        # Determine adapter type
        if adapter_type == "litellm":
            # Force LiteLLM
            return LiteLLMModelAdapter(provider=provider, api_key=api_key)
        
        elif adapter_type == "openai":
            # Force OpenAI
            return OpenAIModelAdapter(provider=provider, api_key=api_key)
        
        elif adapter_type == "auto":
            # Auto-select based on provider compatibility
            # Get provider object if it's a string
            if isinstance(provider, str):
                from ..providers import MODEL_PROVIDERS
                provider_obj = MODEL_PROVIDERS.get(provider.lower())
            else:
                provider_obj = provider
            
            # Check if OpenAI-compatible
            # Providers that support OpenAI adapter: openai, openrouter, groq, etc.
            openai_compatible_providers = {
                "openai", "openrouter", "groq", "xai", "deepseek",
                "cerebras", "cohere", "fireworks", "github", "moonshotai",
                "ollama", "lm_studio"
            }
            
            provider_name = provider_obj.name if provider_obj else provider
            
            if provider_name in openai_compatible_providers or provider_name.startswith("http"):
                _logger.debug(f"Using OpenAI adapter for provider: {provider_name}")
                return OpenAIModelAdapter(provider=provider, api_key=api_key)
            else:
                _logger.debug(f"Using LiteLLM adapter for provider: {provider_name}")
                return LiteLLMModelAdapter(provider=provider, api_key=api_key)
        
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")
    
    @property
    def model(self) -> str:
        """The model name."""
        return self._model
    
    @property
    def adapter(self) -> ModelAdapter:
        """The adapter used by this language model."""
        return self._adapter
    
    @property
    def settings(self) -> LanguageModelSettings | None:
        """Default settings for this language model."""
        return self._settings
    
    async def arun(
        self,
        messages: str | Iterable[ChatCompletionMessageParam],
        *,
        type: Type[T] = str,
        tools : Iterable[ChatCompletionToolParam] | None = None,
        stream: bool = False,
        settings: LanguageModelSettings | None = None,
        **kwargs,
    ) -> LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]:
        """Asynchronously run the language model.
        
        Args:
            messages: Either a string prompt or a list of chat messages
            type: The output type. If `str`, returns raw text. Otherwise,
                generates structured output matching the type (must be a Pydantic model).
            stream: Whether to stream the response
            settings: Settings for this specific invocation (overrides default settings)
            **kwargs: Additional parameters passed to the underlying adapter
        
        Returns:
            LanguageModelResponse or async iterator of responses if streaming
        
        Examples:
            # Simple text completion
            response = await model.arun("What is 2+2?")
            print(response.output)  # "4"
            
            # Structured output
            class Answer(BaseModel):
                result: int
            
            response = await model.arun("What is 2+2?", type=Answer)
            print(response.output.result)  # 4
            
            # Streaming
            async for chunk in model.arun("Write a story", stream=True):
                print(chunk.output, end="")
        """
        _logger.info(
            f"Generating{ 'streamed' if stream else '' } language model response using model: {self._model}.\n"
            f"{'Generating structured output of type: ' + str(type) if type != str else 'Generating chat completion'}"
        )

        # Normalize messages to list format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Add tools to kwargs if provided
        if tools is not None:
            kwargs = kwargs.copy()
            kwargs['tools'] = tools
        
        # Merge settings
        merged_settings = self._merge_settings(settings, kwargs)
        
        # Ensure tools are None when using structured output
        if type != str and merged_settings.get("tools") is not None:
            _logger.warning(
                "Tools parameter is ignored when using structured output (type != str). "
                "Setting tools=None."
            )
            merged_settings["tools"] = None
        
        # Determine if we need structured output
        use_structured_output = type != str
        
        if use_structured_output:
            # Use structured output
            return await self._arun_structured(
                messages=messages,
                response_model=type,
                stream=stream,
                settings=merged_settings,
            )
        else:
            # Use standard chat completion
            return await self._arun_text(
                messages=messages,
                stream=stream,
                settings=merged_settings,
            )
    
    def run(
        self,
        messages: str | Iterable[ChatCompletionMessageParam],
        *,
        type: Type[T] = str,
        tools : Iterable[ChatCompletionToolParam] | None = None,
        stream: bool = False,
        settings: LanguageModelSettings | None = None,
        **kwargs,
    ) -> LanguageModelResponse[T] | Iterator[LanguageModelResponse[T]]:
        """Synchronously run the language model (blocks until complete).
        
        This is a synchronous wrapper around arun(). For async contexts,
        use arun() directly.
        
        Args:
            messages: Either a string prompt or a list of chat messages
            type: The output type. If `str`, returns raw text. Otherwise,
                generates structured output matching the type.
            stream: Whether to stream the response
            settings: Settings for this specific invocation
            **kwargs: Additional parameters
        
        Returns:
            LanguageModelResponse or iterator of responses if streaming
        """
        import asyncio
        
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                if stream:
                    # For streaming, we need to handle the async iterator
                    async def _collect_stream():
                        chunks = []
                        async for chunk in self.arun(
                            messages=messages,
                            type=type,
                            tools=tools,
                            stream=True,
                            settings=settings,
                            **kwargs
                        ):
                            chunks.append(chunk)
                        return chunks
                    
                    chunks = loop.run_until_complete(_collect_stream())
                    return iter(chunks)
                else:
                    return loop.run_until_complete(
                        self.arun(
                            messages=messages,
                            type=type,
                            tools=tools,
                            stream=False,
                            settings=settings,
                            **kwargs
                        )
                    )
            finally:
                loop.close()
        else:
            # Event loop is already running (e.g., in Jupyter)
            # We can't use run_until_complete, so raise an error
            raise RuntimeError(
                "Cannot use run() when an event loop is already running. "
                "Use await arun() instead in async contexts."
            )
    
    async def _arun_text(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        stream: bool,
        settings: dict,
    ) -> LanguageModelResponse[str] | AsyncIterator[LanguageModelResponse[str]]:
        """Internal method for standard text completions."""
        
        # Clean model name (remove provider prefix if present)
        model_name = self._adapter.provider.clean_model(self._model)
        
        if stream:
            async def _stream_wrapper():
                chunk_iterator = await self._adapter.create_chat_completion(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    **settings
                )
                
                async for chunk in chunk_iterator:
                    # Extract content from chunk
                    content = None
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            content = delta.content
                    
                    yield LanguageModelResponse[str](
                        output=content,
                        completion=chunk,
                        instructor_mode=None,
                    )
            
            return _stream_wrapper()
        else:
            completion = await self._adapter.create_chat_completion(
                model=model_name,
                messages=messages,
                stream=False,
                **settings
            )
            
            # Extract content from completion
            content = None
            if completion.choices and len(completion.choices) > 0:
                message = completion.choices[0].message
                if hasattr(message, 'content') and message.content:
                    content = message.content
            
            return LanguageModelResponse[str](
                output=content,
                completion=completion,
                instructor_mode=None,
            )
    
    async def _arun_structured(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        response_model: Type[T],
        stream: bool,
        settings: dict,
    ) -> LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]:
        """Internal method for structured output generation."""
        _logger.debug(
            f"Running structured output with model: {self._model}, "
            f"response_model: {response_model.__name__}, stream: {stream}"
        )
        
        # Get instructor mode from settings or use default
        instructor_mode = settings.pop("instructor_mode", None)
        
        # Clean model name
        model_name = self._adapter.provider.clean_model(self._model)
        
        if stream:
            async def _stream_wrapper():
                result_iterator = await self._adapter.create_structured_output(
                    model=model_name,
                    messages=messages,
                    response_model=response_model,
                    instructor_mode=instructor_mode,
                    stream=True,
                    **settings
                )
                
                async for output, completion in result_iterator:
                    yield LanguageModelResponse[T](
                        output=output,
                        completion=completion,
                        instructor_mode=instructor_mode or self._adapter.instructor_mode,
                    )
            
            return _stream_wrapper()
        else:
            output, completion = await self._adapter.create_structured_output(
                model=model_name,
                messages=messages,
                response_model=response_model,
                instructor_mode=instructor_mode,
                stream=False,
                **settings
            )
            
            return LanguageModelResponse[T](
                output=output,
                completion=completion,
                instructor_mode=instructor_mode or self._adapter.instructor_mode,
            )
    
    def _merge_settings(
        self,
        invocation_settings: LanguageModelSettings | None,
        kwargs: dict,
    ) -> dict:
        """Merge default settings, invocation settings, and kwargs.
        
        Priority (highest to lowest):
        1. kwargs
        2. invocation_settings
        3. default settings (self._settings)
        
        Args:
            invocation_settings: Settings provided for this specific invocation
            kwargs: Additional keyword arguments
        
        Returns:
            Merged settings dictionary
        """
        merged = {}
        
        # Start with default settings
        if self._settings:
            merged.update(self._settings.model_dump(exclude_none=True))
        
        # Override with invocation settings
        if invocation_settings:
            merged.update(invocation_settings.model_dump(exclude_none=True))
        
        # Override with kwargs
        merged.update(kwargs)
        
        return merged