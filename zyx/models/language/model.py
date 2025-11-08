"""zyx.models.language.model"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import asdict
from typing import Any, Dict, Generic, Literal, Type, TypeVar

from instructor import Mode
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from ...core.schemas.schema import Schema
from ..definition import ModelDefinition, ModelDefinitionError
from ..providers import ModelProvider, ModelProviderName
from .types import LanguageModelName, LanguageModelResponse, LanguageModelSettings

__all__ = ["LanguageModel", "arun_llm", "run_llm", "llm"]


_logger = logging.getLogger(__name__)


T = TypeVar("T")


class LanguageModel(ModelDefinition, Generic[T]):
    """A simple, unified interface around language model provider clients,
    (OpenAI, LiteLLM) along with the Instructor library for generating chat
    completions and structured outputs.

    !!! note
        This class is generic over the type of the response.

    All `Model` suffix class interfaces within `ZYX` (LanguageModel,
    EmbeddingModel, etc.) implement a `run()` and `arun()` method for
    all 'unified' or main functionality."""

    @property
    def kind(self) -> str:
        return "language_model"

    @property
    def model(self) -> LanguageModelName | str:
        """Get the model name for this language model."""
        return self._model

    @property
    def settings(self) -> LanguageModelSettings:
        """Get the settings for this language model."""
        return self._settings

    def __init__(
        self,
        model: LanguageModelName | str = "openai/gpt-4o-mini",
        *,
        provider: ModelProviderName | ModelProvider | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        settings: LanguageModelSettings | None = None,
    ) -> None:
        """Initialize the language model with a model name, and optional provider
        name / custom base URL and API key, along with an optional set of
        settings & tools to use as default for this model.

        NOTE: A `LanguageModel` instance does not run tools.
        """
        super().__init__(
            model=model,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            settings=settings,
        )

    async def _arun_stream(
        self, params: Dict[str, Any]
    ) -> AsyncIterator[LanguageModelResponse[T]]:
        if params.get("type") is not str:
            async for chunk in await self.get_client().structured_output(
                model=params["model"],
                messages=params["messages"],
                response_model=params["type"],
                instructor_mode=params.get("instructor_mode"),
                stream=True,
                **params.get("kwargs", {}),
            ):
                yield LanguageModelResponse(
                    raw=chunk.get("raw"),
                    instructor_mode=chunk.get("instructor_mode"),
                    delta=chunk.get("output"),
                )
        else:
            async for chunk in await self.get_client().chat_completion(
                model=params["model"],
                messages=params["messages"],
                stream=True,
                **params.get("kwargs", {}),
            ):
                yield LanguageModelResponse(
                    raw=chunk,
                    content=chunk.choices[0].delta.content,
                )

    async def _arun_non_stream(
        self, params: Dict[str, Any]
    ) -> LanguageModelResponse[T]:
        if params.get("type") is not str:
            response = await self.get_client().structured_output(
                model=params["model"],
                messages=params["messages"],
                response_model=params["type"],
                instructor_mode=params.get("instructor_mode"),
                **params.get("kwargs", {}),
            )
            return LanguageModelResponse(
                raw=response.get("raw"),
                instructor_mode=response.get("instructor_mode"),
                content=response.get("output"),
            )
        else:
            response = await self.get_client().chat_completion(
                model=params["model"],
                messages=params["messages"],
                **params.get("kwargs", {}),
            )
            return LanguageModelResponse(
                raw=response,
                content=response.choices[0].message.content,
            )

    async def arun(
        self,
        messages: Iterable[ChatCompletionMessageParam] | str,
        *,
        type: Type[T] | Schema[T] = str,
        instructions: str | None = None,
        title: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
        key: str | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        stream: bool = False,
        instructor_mode: Mode | None = None,
        settings: LanguageModelSettings | None = None,
    ) -> LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]:
        """Asyncronously run a language model with a given set of messages, a response
        type, and optional instructions, tools, and settings.

        Parameters
        ----------
        messages : str | Iterable[ChatCompletionMessageParam]
            The messages to send to the language model.
        type : Type[T]
            The type of the response to generate.
        instructions : str | None
            The instructions to send to the language model.
        title : str | None
            The title of the schema to use for the response.
        description : str | None
            The description of the schema to use for the response.
        exclude : set[str] | None
            The fields to exclude from the schema to use for the response.
        key : str | None
            For types such as int, Literal, etc. Instructor assigns the key of the field to use
            the name `content`. You can override this value for these simple type cases
            by providing a custom key.
        tools : Iterable[ChatCompletionToolParam] | None
            The tools to use for the language model.
        stream : bool
            Whether to stream the response from the language model.
        instructor_mode : Mode | None
            The instructor mode to use for the language model.
        settings : LanguageModelSettings | None
            The settings to use for the language model.

        Returns
        -------
        LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]
            The response from the language model.
        """
        if not self._provider:
            raise ValueError("Cannot run model without a provider")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if instructions:
            messages.insert(0, {"role": "system", "content": instructions})

        try:
            # Merge settings with defaults
            if settings is None:
                merged_settings = self.settings
            else:
                # Get base settings as dict
                base_settings_dict = asdict(self.settings) if self.settings else {}
                # Get override settings as dict
                settings_dict = asdict(settings) if settings else {}
                # Merge the two dicts, with settings_dict taking precedence
                merged_dict = {**base_settings_dict, **settings_dict}
                # Create new LanguageModelSettings instance from merged dict
                merged_settings = LanguageModelSettings(**merged_dict)

            # Filter out None values to avoid passing None params to API
            merged_settings_dict = (
                {k: v for k, v in merged_settings.__dict__.items() if v is not None}
                if merged_settings
                else {}
            )
        except Exception as e:
            raise ModelDefinitionError(
                f"Error merging settings: {e}",
                model=self._model,
            ) from e

        if tools:
            if type is not str:
                raise ValueError(
                    f"Cannot provide tools for a structured output, passed {len(tools)} for output of type: {type.__name__}"
                )

            merged_settings_dict["tools"] = tools

        if type is not str:
            # !! check for schema
            if isinstance(type, Schema):
                type = type.pydantic_model

            # build response object
            if any([title, description, exclude]):
                type = Schema(
                    type, title=title, description=description, exclude=exclude
                ).pydantic_model

            params = {
                "model": self._provider.clean_model_name(self._model),
                "messages": messages,
                "type": type,
                "instructor_mode": instructor_mode,
                "kwargs": merged_settings_dict,
            }
        else:
            params = {
                "model": self._provider.clean_model_name(self._model),
                "messages": messages,
                "type": type,
                "kwargs": merged_settings_dict,
            }

        if stream:
            return await self._arun_stream(params)
        else:
            return await self._arun_non_stream(params)

    def run(
        self,
        messages: Iterable[ChatCompletionMessageParam] | str,
        *,
        type: Type[T] | Schema[T] = str,
        instructions: str | None = None,
        title: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        stream: bool = False,
        instructor_mode: Mode | None = None,
        settings: LanguageModelSettings | None = None,
    ) -> LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]:
        """Run a language model with a given set of messages, a response
        type, and optional instructions, tools, and settings.

        Parameters
        ----------
        messages : str | Iterable[ChatCompletionMessageParam]
            The messages to send to the language model.
        type : Type[T]
            The type of the response to generate.
        instructions : str | None
            The instructions to send to the language model.
        tools : Iterable[ChatCompletionToolParam] | None
            The tools to use for the language model.
        stream : bool
            Whether to stream the response from the language model.
        instructor_mode : Mode | None
            The instructor mode to use for the language model.
        settings : LanguageModelSettings | None
            The settings to use for the language model.

        Returns
        -------
        LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]
            The response from the language model.
        """
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.arun(
                        messages=messages,
                        type=type,
                        instructions=instructions,
                        title=title,
                        description=description,
                        exclude=exclude,
                        tools=tools,
                        stream=stream,
                        instructor_mode=instructor_mode,
                        settings=settings,
                    ),
                )
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(
                self.arun(
                    messages=messages,
                    type=type,
                    instructions=instructions,
                    title=title,
                    description=description,
                    exclude=exclude,
                    tools=tools,
                    stream=stream,
                    instructor_mode=instructor_mode,
                    settings=settings,
                )
            )


async def arun_llm(
    messages: str | Iterable[ChatCompletionMessageParam],
    model: LanguageModelName | str = "openai/gpt-4o-mini",
    type: Type[T] | Schema[T] = str,
    *,
    provider: ModelProviderName | ModelProvider | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    instructions: str | None = None,
    title: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
    tools: Iterable[ChatCompletionToolParam] | None = None,
    instructor_mode: Mode | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_logprobs: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    tool_choice: Literal["auto", "required", "none"] | str | None = None,
    parallel_tool_calls: bool | None = None,
    stream: bool = False,
) -> LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]:
    """Asynchronously run a language model with a given set of messages, a response
    type, and optional instructions, tools, and settings.

    Parameters
    ----------
    messages : str | Iterable[ChatCompletionMessageParam]
        The messages to send to the language model. This can be a string, or an iterable of ChatCompletionMessageParam.
    model : LanguageModelName | str
        The model name to use. This should use the provider name prefix if applicable. (ex: "openai/gpt-4o-mini")
    type : Type[T] | Schema[T]
        The type of the response to generate.
    provider : ModelProviderName | ModelProvider | None
        The provider name or instance. If not provided, the provider will be inferred from the model name.
    base_url : str | None
        Custom base URL for the model provider. Cannot be used with provider parameter.
    api_key : str | None
        API key for the model provider. If not provided, will use environment variable.
    instructions : str | None
        Additional instructions to guide the model's response.
    title : str | None
        Title for the structured output schema.
    description : str | None
        Description for the structured output schema.
    exclude : set[str] | None
        Set of field names to exclude from the structured output.
    tools : Iterable[ChatCompletionToolParam] | None
        Tools available for the model to call.
    instructor_mode : Mode | None
        Instructor mode for structured output generation.
    max_tokens : int | None
        Maximum number of tokens to generate.
    temperature : float | None
        Sampling temperature (0.0 to 2.0).
    top_p : float | None
        Nucleus sampling parameter.
    top_logprobs : int | None
        Number of most likely tokens to return at each position.
    frequency_penalty : float | None
        Penalty for token frequency (-2.0 to 2.0).
    presence_penalty : float | None
        Penalty for token presence (-2.0 to 2.0).
    tool_choice : Literal["auto", "required", "none"] | str | None
        Strategy for tool selection.
    parallel_tool_calls : bool | None
        Whether to allow parallel tool calls.
    stream : bool
        Whether to stream the response.


    Returns
    -------
    LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]
        The response from the language model.
    """
    return await LanguageModel(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        settings=LanguageModelSettings(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_logprobs,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        ),
    ).arun(
        messages=messages,
        type=type,
        instructions=instructions,
        title=title,
        description=description,
        exclude=exclude,
        tools=tools,
        stream=stream,
        instructor_mode=instructor_mode,
    )


def run_llm(
    messages: str | Iterable[ChatCompletionMessageParam],
    model: LanguageModelName | str = "openai/gpt-4o-mini",
    type: Type[T] | Schema[T] = str,
    *,
    provider: ModelProviderName | ModelProvider | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    instructions: str | None = None,
    title: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
    tools: Iterable[ChatCompletionToolParam] | None = None,
    instructor_mode: Mode | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_logprobs: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    tool_choice: Literal["auto", "required", "none"] | str | None = None,
    parallel_tool_calls: bool | None = None,
    stream: bool = False,
) -> LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]:
    """Run a language model with a given set of messages, a response
    type, and optional instructions, tools, and settings.

    Parameters
    ----------
    messages : str | Iterable[ChatCompletionMessageParam]
        The messages to send to the language model. This can be a string, or an iterable of ChatCompletionMessageParam.
    model : LanguageModelName | str
        The model name to use. This should use the provider name prefix if applicable. (ex: "openai/gpt-4o-mini")
    type : Type[T] | Schema[T]
        The type of the response to generate.
    provider : ModelProviderName | ModelProvider | None
        The provider name or instance. If not provided, the provider will be inferred from the model name.
    base_url : str | None
        Custom base URL for the model provider. Cannot be used with provider parameter.
    api_key : str | None
        API key for the model provider. If not provided, will use environment variable.
    instructions : str | None
        Additional instructions to guide the model's response.
    title : str | None
        Title for the structured output schema.
    description : str | None
        Description for the structured output schema.
    exclude : set[str] | None
        Set of field names to exclude from the structured output.
    tools : Iterable[ChatCompletionToolParam] | None
        Tools available for the model to call.
    instructor_mode : Mode | None
        Instructor mode for structured output generation.
    max_tokens : int | None
        Maximum number of tokens to generate.
    temperature : float | None
        Sampling temperature (0.0 to 2.0).
    top_p : float | None
        Nucleus sampling parameter.
    top_logprobs : int | None
        Number of most likely tokens to return at each position.
    frequency_penalty : float | None
        Penalty for token frequency (-2.0 to 2.0).
    presence_penalty : float | None
        Penalty for token presence (-2.0 to 2.0).
    tool_choice : Literal["auto", "required", "none"] | str | None
        Strategy for tool selection.
    parallel_tool_calls : bool | None
        Whether to allow parallel tool calls.
    stream : bool
        Whether to stream the response.

    Returns
    -------
    LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]
        The response from the language model.
    """
    return LanguageModel(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        settings=LanguageModelSettings(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_logprobs,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        ),
    ).run(
        messages=messages,
        type=type,
        instructions=instructions,
        title=title,
        description=description,
        exclude=exclude,
        tools=tools,
        stream=stream,
        instructor_mode=instructor_mode,
    )


def llm(
    type: Type[T] | Schema[T],
    model: LanguageModelName | str = "openai/gpt-4o-mini",
    *,
    instructions: str | None = None,
    title: str | None = None,
    description: str | None = None,
    exclude: set[str] | None = None,
    provider: ModelProviderName | ModelProvider | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_logprobs: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    tool_choice: Literal["auto", "required", "none"] | str | None = None,
    parallel_tool_calls: bool | None = None,
) -> LanguageModel[T]:
    """Create a language model locked to a specific response type.

    !!! note
    This function requires the response type T to be provided.

    Parameters
    ----------
    type : Type[T] | Schema[T]
        The type of the response to generate.
    model : LanguageModelName | str
        The model name to use.
    provider : ModelProviderName | ModelProvider | None
        The provider name or instance.
    instructions : str | None
        Additional instructions to guide the model's response.
    title : str | None
        Title for the structured output schema.
    description : str | None
        Description for the structured output schema.
    exclude : set[str] | None
        Set of field names to exclude from the structured output.
    base_url : str | None
        Custom base URL for the provider.
    api_key : str | None
        API key for the provider.
    settings : LanguageModelSettings | None
        Default settings for the language model.
    max_tokens : int | None
        Maximum number of tokens to generate.
    temperature : float | None
        Sampling temperature (0.0 to 2.0).
    top_p : float | None
        Nucleus sampling parameter.
    top_logprobs : int | None
        Number of most likely tokens to return at each position.
    frequency_penalty : float | None
        Penalty for token frequency (-2.0 to 2.0).
    presence_penalty : float | None
        Penalty for token presence (-2.0 to 2.0).
    tool_choice : Literal["auto", "required", "none"] | str | None
        Strategy for tool selection.
    parallel_tool_calls : bool | None
        Whether to allow parallel tool calls.

    Returns
    -------
    LanguageModel[T]
        A LanguageModel instance with run and arun methods pre-bound to the specified type.

    Examples
    --------
        ```python
        >>> my_llm = llm(int, model="openai/gpt-4o-mini")
        >>> response = my_llm.run("What is 2 + 2?")
        >>> print(response.content)
        ```

        ```bash title="Output"
        4
        ```
    """
    model_instance: LanguageModel[T] = LanguageModel(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        settings=LanguageModelSettings(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_logprobs,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        ),
    )
    # Bind the type to run and arun methods
    model_instance.run = functools.partial(
        model_instance.run,
        instructions=instructions,
        type=type,
        title=title,
        description=description,
        exclude=exclude,
    )
    model_instance.arun = functools.partial(
        model_instance.arun,
        instructions=instructions,
        type=type,
        title=title,
        description=description,
        exclude=exclude,
    )
    return model_instance
