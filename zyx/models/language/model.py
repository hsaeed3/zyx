"""zyx.models.language.model"""

from __future__ import annotations

import logging
import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import asdict
from typing import (
    Any,
    Dict,
    TypeVar,
    Type,
)

from instructor import Mode
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from ..providers import (
    ModelProvider,
    ModelProviderName,
    ModelProviderRegistry,
)
from ..clients import ModelClient
from ..clients.openai import OpenAIModelClient
from ..clients.litellm import LiteLLMModelClient
from .types import LanguageModelSettings, LanguageModelResponse, LanguageModelName

__all__ = [
    "LanguageModel",
]


_logger = logging.getLogger(__name__)


T = TypeVar("T")


class LanguageModel:
    """A simple, unified interface around language model provider clients,
    (OpenAI, LiteLLM) along with the Instructor library for generating chat
    completions and structured outputs.

    All `Model` suffix class interfaces within `ZYX` (LanguageModel,
    EmbeddingModel, etc.) implement a `run()` and `arun()` method for
    all 'unified' or main functionality."""

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
        self._model = model
        self._client: ModelClient[T] | None = None
        self._provider: ModelProvider | None = None
        self._settings = settings or LanguageModelSettings()

        if base_url:
            assert not provider, "Cannot provide base_url and a provider"

            # try to get custom provider if exists, all custom providers
            # use an OpenAI client
            if ModelProviderRegistry().get(f"custom:{base_url}"):
                self._provider = ModelProviderRegistry().get(f"custom:{base_url}")
            else:
                self._provider = ModelProviderRegistry().register_custom(
                    base_url, api_key
                )

        if provider:
            if isinstance(provider, str):
                self._provider = ModelProviderRegistry().get(provider)

                if not self._provider:
                    raise ValueError(f"Unknown provider: {provider}")

            elif isinstance(provider, ModelProvider):
                self._provider = provider
            else:
                raise ValueError(f"Invalid provider: {provider}")

        else:
            self._provider = ModelProviderRegistry().infer_from_model_name(
                model=self._model,
                kind="language_model",
            )

            if not self._provider:
                # this will fallback to the LiteLLM client
                self._provider = ModelProvider(name="unknown")

    def __str__(self):
        from ..._internal._beautification import _pretty_print_model

        return _pretty_print_model(self, "language_model")

    def __rich__(self):
        from ..._internal._beautification import _rich_pretty_print_model

        return _rich_pretty_print_model(self, "language_model")

    @property
    def model(self) -> LanguageModelName | str:
        """Get the model name for this language model."""
        return self._model

    @property
    def settings(self) -> LanguageModelSettings:
        """Get the settings for this language model."""
        return self._settings

    def get_client(self) -> ModelClient[T]:
        """Get the inferred / associated model client for this language model.

        This requires the `_provider` attribute to be set as a ModelProvider
        object."""

        if not self._provider:
            raise ValueError("Cannot get client without a provider")

        if not self._client:
            if "openai" in self._provider.supported_clients:
                self._client = OpenAIModelClient(
                    base_url=self._provider.base_url,
                    api_key=self._provider.get_api_key(),
                )
            else:
                self._client = LiteLLMModelClient(
                    base_url=self._provider.base_url,
                    api_key=self._provider.get_api_key(),
                )

        return self._client

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
                    content=chunk.choices[0].message.content,
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
        type: Type[T] = str,
        instructions: str | None = None,
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

        merged_settings = (
            self.settings
            if settings is None
            else LanguageModelSettings(**{**asdict(self.settings), **asdict(settings)})
        )
        # Filter out None values from settings to avoid passing None params to API
        merged_settings_dict = {
            k: v for k, v in asdict(merged_settings).items() if v is not None
        }

        if tools:
            if type is not str:
                raise ValueError(
                    f"Cannot provide tools for a structured output, passed {len(tools)} for output of type: {type.__name__}"
                )

            merged_settings_dict["tools"] = tools

        if type is not str:
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
        type: Type[T] = str,
        instructions: str | None = None,
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
                    tools=tools,
                    stream=stream,
                    instructor_mode=instructor_mode,
                    settings=settings,
                )
            )
