"""zyx.core.models.language.model"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import asdict
from typing import Any, Dict, Generic, List, Type, TypeVar

from instructor import Mode
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from ...processing.schemas.schema import Schema
from ..definition import ModelDefinition, ModelDefinitionError
from ..providers import ModelProvider, ModelProviderName
from .types import (
    LanguageModelName,
    LanguageModelResponse,
    LanguageModelSettings,
)

__all__ = ["LanguageModel"]


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
    def type(self) -> Type[T] | Schema[T]:
        """The default response type for this language model."""
        return self._type

    @property
    def model(self) -> LanguageModelName | str:
        """Get the model name for this language model."""
        return self._model

    @property
    def settings(self) -> LanguageModelSettings:
        """Get the settings for this language model."""
        return self._settings

    @property
    def instructions(self) -> str | None:
        """The instructions for this language model."""
        return self._instructions

    @instructions.setter
    def instructions(self, instructions: str | None) -> None:
        """Set the instructions for this language model."""
        self._instructions = instructions

    def __init__(
        self,
        model: LanguageModelName | str = "openai/gpt-4o-mini",
        type: Type[T] | Schema[T] = str,
        *,
        instructions: str | None = None,
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
        self._type = type
        self._instructions = instructions

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
        type: Type[T] | Schema[T] | None = None,
        instructions: str | None = None,
        title: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
        key: str | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        stream: bool = False,
        instructor_mode: Mode | None = None,
        settings: LanguageModelSettings | None = None,
    ) -> (
        LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]
    ):
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
        if type is None:
            type = self.type

        if instructions:
            if self.instructions:
                instructions = f"{self.instructions}\n\n{instructions}"
        else:
            instructions = self.instructions

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
                base_settings_dict = (
                    asdict(self.settings) if self.settings else {}
                )
                # Get override settings as dict
                settings_dict = asdict(settings) if settings else {}
                # Merge the two dicts, with settings_dict taking precedence
                merged_dict = {**base_settings_dict, **settings_dict}
                # Create new LanguageModelSettings instance from merged dict
                merged_settings = LanguageModelSettings(**merged_dict)

            # Filter out None values to avoid passing None params to API
            merged_settings_dict = (
                {
                    k: v
                    for k, v in merged_settings.__dict__.items()
                    if v is not None
                }
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
                type = type.model

            # build response object
            if any([title, description, exclude]):
                type = Schema(
                    type,
                    title=title,
                    description=description,
                    exclude=exclude,
                    key=key,
                ).model

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
        type: Type[T] | Schema[T] | None = None,
        instructions: str | None = None,
        title: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        stream: bool = False,
        instructor_mode: Mode | None = None,
        settings: LanguageModelSettings | None = None,
    ) -> (
        LanguageModelResponse[T] | AsyncIterator[LanguageModelResponse[T]]
    ):
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
        if type is None:
            type = self.type

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

    async def abatch_run(
        self,
        batch_messages: List[Iterable[ChatCompletionMessageParam] | str],
        *,
        type: Type[T] | Schema[T] | None = None,
        instructions: str | None = None,
        title: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
        key: str | None = None,
        instructor_mode: Mode | None = None,
        settings: LanguageModelSettings | None = None,
    ) -> List[LanguageModelResponse[T]]:
        """Asyncronously run multiple language model requests concurrently.

        Parameters
        ----------
        batch_messages : List[str | Iterable[ChatCompletionMessageParam]]
            A list of message sets to send to the language model.
        type : Type[T] | Schema[T] | None
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
        instructor_mode : Mode | None
            The instructor mode to use for the language model.
        settings : LanguageModelSettings | None
            The settings to use for the language model.

        Returns
        -------
        List[LanguageModelResponse[T]]
            A list of responses from the language model in the same order as the input.
        """
        if type is None:
            type = self.type

        if instructions:
            if self.instructions:
                instructions = f"{self.instructions}\n\n{instructions}"
        else:
            instructions = self.instructions

        if not self._provider:
            raise ValueError("Cannot run model without a provider")

        # Process all message lists
        processed_messages = []
        for messages in batch_messages:
            if isinstance(messages, str):
                msg_list = [{"role": "user", "content": messages}]
            else:
                msg_list = list(messages)

            if instructions:
                msg_list.insert(
                    0, {"role": "system", "content": instructions}
                )

            processed_messages.append(msg_list)

        try:
            # Merge settings with defaults
            if settings is None:
                merged_settings = self.settings
            else:
                # Get base settings as dict
                base_settings_dict = (
                    asdict(self.settings) if self.settings else {}
                )
                # Get override settings as dict
                settings_dict = asdict(settings) if settings else {}
                # Merge the two dicts, with settings_dict taking precedence
                merged_dict = {**base_settings_dict, **settings_dict}
                # Create new LanguageModelSettings instance from merged dict
                merged_settings = LanguageModelSettings(**merged_dict)

            # Filter out None values to avoid passing None params to API
            merged_settings_dict = (
                {
                    k: v
                    for k, v in merged_settings.__dict__.items()
                    if v is not None
                }
                if merged_settings
                else {}
            )
        except Exception as e:
            raise ModelDefinitionError(
                f"Error merging settings: {e}",
                model=self._model,
            ) from e

        if type is not str:
            # !! check for schema
            if isinstance(type, Schema):
                type = type.model

            # build response object
            if any([title, description, exclude]):
                type = Schema(
                    type,
                    title=title,
                    description=description,
                    exclude=exclude,
                    key=key,
                ).model

            _logger.debug(
                f"LanguageModel: Generating {len(batch_messages)} concurrent structured outputs with model {self._model}."
            )

            results = await self.get_client().batch_structured_output(
                model=self._provider.clean_model_name(self._model),
                messages=processed_messages,
                response_model=type,
                instructor_mode=instructor_mode,
                **merged_settings_dict,
            )

            return [
                LanguageModelResponse(
                    raw=result.get("raw"),
                    instructor_mode=result.get("instructor_mode"),
                    content=result.get("output"),
                )
                for result in results
            ]
        else:
            _logger.debug(
                f"LanguageModel: Generating {len(batch_messages)} concurrent chat completions with model {self._model}."
            )

            results = await self.get_client().batch_chat_completion(
                model=self._provider.clean_model_name(self._model),
                messages=processed_messages,
                **merged_settings_dict,
            )

            return [
                LanguageModelResponse(
                    raw=result,
                    content=result.choices[0].message.content,
                )
                for result in results
            ]

    def batch_run(
        self,
        batch_messages: List[Iterable[ChatCompletionMessageParam] | str],
        *,
        type: Type[T] | Schema[T] | None = None,
        instructions: str | None = None,
        title: str | None = None,
        description: str | None = None,
        exclude: set[str] | None = None,
        instructor_mode: Mode | None = None,
        settings: LanguageModelSettings | None = None,
    ) -> List[LanguageModelResponse[T]]:
        """Run multiple language model requests concurrently.

        Parameters
        ----------
        batch_messages : List[str | Iterable[ChatCompletionMessageParam]]
            A list of message sets to send to the language model.
        type : Type[T] | Schema[T] | None
            The type of the response to generate.
        instructions : str | None
            The instructions to send to the language model.
        title : str | None
            The title of the schema to use for the response.
        description : str | None
            The description of the schema to use for the response.
        exclude : set[str] | None
            The fields to exclude from the schema to use for the response.
        instructor_mode : Mode | None
            The instructor mode to use for the language model.
        settings : LanguageModelSettings | None
            The settings to use for the language model.

        Returns
        -------
        List[LanguageModelResponse[T]]
            A list of responses from the language model in the same order as the input.
        """
        if type is None:
            type = self.type

        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.abatch_run(
                        batch_messages=batch_messages,
                        type=type,
                        instructions=instructions,
                        title=title,
                        description=description,
                        exclude=exclude,
                        instructor_mode=instructor_mode,
                        settings=settings,
                    ),
                )
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(
                self.abatch_run(
                    batch_messages=batch_messages,
                    type=type,
                    instructions=instructions,
                    title=title,
                    description=description,
                    exclude=exclude,
                    instructor_mode=instructor_mode,
                    settings=settings,
                )
            )
