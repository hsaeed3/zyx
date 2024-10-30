# zyx.completions.base_client
# Base LLM client
# Handles all LLM completions

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import Field
    from typing import Any, Callable, Dict, List, Optional, Type, Literal

__all__ = ["Client", "completion"]

# Standard library imports
import json
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Type,
    Union,
)

# Third-party imports
import httpx
import openai
from pydantic import BaseModel, create_model
from rich.progress import Progress

# types
from ..resources.types.config.client import (
    ClientConfig, ClientProvider, ClientChecks,
    DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
)
from ..resources.types.completions.arguments import (
    ChatCompletion, Completion, Tool, ToolType, CompletionArguments, Message,
    InstructorMode, FunctionCall, Process, ChatCompletionModality,
    ResponseFormat, ChatCompletionAudioParam, ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam, ChatModel, Function
)
from ..resources.types.model_outputs import (
    StringResponse, IntResponse, FloatResponse, BoolResponse, ListResponse
)

# Core utils
from ..lib.exceptions import Yikes, ZyxError
from ..lib.utils import console, logger
from ..lib.environment import ZYX_DEFAULT_MODEL
from ..resources.utils import function_calling
from ..resources.utils.tool_generator import _generate_tool

# LiteLLM Generic
LiteLLM = Type["LiteLLM"]


class Recommendation(BaseModel):
    """Internal recommendation class."""

    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class Client:
    """Base LLM Completions Client for Zyx."""

    def __init__(
        self,
        provider: Union[Literal["openai", "litellm"], ClientProvider] = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = DEFAULT_TIMEOUT,
        max_retries: Optional[int] = DEFAULT_MAX_RETRIES,
        default_headers: Optional[Mapping[str, str]] = None,
        default_query: Optional[Mapping[str, object]] = None,
        verify_ssl: Optional[bool] = None,
        http_args: Optional[Mapping[str, object]] = None,
        http_client: Optional[httpx.Client] = None,
        verbose: Optional[bool] = None,
    ):
        """Initialize a Base LLM Completions Client for Zyx.

        Args:
            provider(Union[Literal["openai", "litellm"], ClientProvider]): The provider to use for completions.
            api_key(Optional[str]): The API key to use for completions.
            base_url(Optional[str]): The base URL to use for completions.
            organization(Optional[str]): The organization to use for completions.
            project(Optional[str]): The project to use for completions.
            timeout(Optional[Union[float, httpx.Timeout]]): The timeout to use for completions.
            max_retries(Optional[int]): The maximum number of retries to use for completions.
            default_headers(Optional[Mapping[str, str]]): The default headers to use for completions.
            default_query(Optional[Mapping[str, object]]): The default query parameters to use for completions.
            verify_ssl(Optional[bool]): Whether to verify SSL certificates.
            http_args(Optional[Mapping[str, object]]): The HTTP arguments to use for completions.
            http_client(Optional[httpx.Client]): The HTTP client to use for completions.
            verbose(Optional[bool]): Whether to print verbose output.
        """
        # Initialize client config
        self.config = ClientConfig(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            verify_ssl=verify_ssl,
            http_args=http_args,
            http_client=http_client,
            verbose=verbose,
        )

        # Initialize client checks
        self.checks = ClientChecks()

        # Initialize client
        self._init_client()

    # ----------------------------------------------------------------
    # RECOMMENDATION
    # ----------------------------------------------------------------
    @staticmethod
    def _recommend(
        model: str, base_url: Optional[str] = None, api_key: Optional[str] = None
    ) -> Recommendation:
        """Recommends args based on the given model name."""
        if base_url:
            return Recommendation(
                provider="openai", model=model, base_url=base_url, api_key=api_key
            )

        rec = Recommendation()

        if model.startswith(("gpt-", "o1", "openai/", "chatgpt-", "ollama/")):
            rec.provider = "openai"

            if model.startswith(("openai/", "chatgpt-", "o1", "gpt-")):
                if model.startswith("openai/"):
                    model = model[7:]
                else:
                    rec.model = model
                return rec

            elif model.startswith("ollama/"):
                rec.model = model[7:]
                rec.base_url = "http://localhost:11434/v1"
                rec.api_key = api_key if api_key else "ollama"
                return rec
        else:
            rec.model = model
            rec.api_key = api_key
            rec.provider = "litellm"
            return rec

    # ----------------------------------------------------------------
    # CLIENT INITIALIZATION
    # ----------------------------------------------------------------

    def __get_openai_client(self) -> openai.OpenAI:
        """Returns the OpenAI client."""
        from openai import OpenAI

        try:
            return OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization,
                project=self.config.project,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=self.config.default_headers,
                default_query=self.config.default_query,
                http_client=self.config.http_client,
            )
        except Exception as e:
            raise ZyxError(
                f"❌ [bold dark_red]ClientInitializationError[/bold dark_red] Failed to initialize [bold]OpenAI[/bold] client: {e}"
            )

    def __get_litellm_client(self) -> LiteLLM:
        """Returns the LiteLLM client."""
        from litellm import LiteLLM

        try:
            return LiteLLM(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=self.config.default_headers,
            )
        except Exception as e:
            raise ZyxError(
                f"❌ [bold dark_red]ClientInitializationError[/bold dark_red] Failed to initialize [bold]LiteLLM[/bold] client: {e}"
            )

    def _init_client(self) -> Union[openai.OpenAI, LiteLLM]:
        """Initializes the client based on the provider."""
        if self.config.provider == "openai":
            if self.config.verify_ssl is not None or self.config.http_args is not None:
                if self.config.http_client is not None:
                    logger.warning(
                        "⚠️ [bold]Verify SSL[/bold] or [bold]HTTP args[/bold] detected, any passed http client will be ignored."
                    )
                self.config.http_client = httpx.Client(
                    verify=self.config.verify_ssl, **self.config.http_args
                )
            self._base_client = self.__get_openai_client()
        else:
            self._base_client = self.__get_litellm_client()

        self.checks.is_client_initialized = True

        if self.config.verbose:
            console.print(f"✅ Initialized {self.config.provider} client.")

    # ----------------------------------------------------------------
    # INSTRUCTOR PATCHES
    # ----------------------------------------------------------------

    def __instructor_patch_openai_client(
        self, mode: InstructorMode = "tool_call"
    ) -> openai.OpenAI:
        """Patches the OpenAI client with instructor & given mode."""
        from instructor import Mode, from_openai

        try:
            return from_openai(self._base_client, mode=Mode(mode))
        except Exception as e:
            raise ZyxError(
                f"❌ [bold dark_red]InstructorPatchError[/bold dark_red] Failed to patch [bold]OpenAI[/bold] client: {e}"
            )

    def __instructor_patch_litellm_client(
        self, mode: InstructorMode = "tool_call"
    ) -> LiteLLM:
        """Patches the LiteLLM client with instructor."""
        from instructor import Mode, patch as litellm_patch

        try:
            return litellm_patch(self._base_client, mode=Mode(mode))
        except Exception as e:
            raise ZyxError(
                f"❌ [bold dark_red]InstructorPatchError[/bold dark_red] Failed to patch [bold]LiteLLM[/bold] client: {e}"
            )

    def _patch_client(
        self, mode: InstructorMode = "tool_call"
    ) -> Union[openai.OpenAI, LiteLLM]:
        """Patches the client with instructor."""
        self.instructor = None

        if self.config.provider == "openai":
            self.instructor = self.__instructor_patch_openai_client(mode)
        else:
            self.instructor = self.__instructor_patch_litellm_client(mode)

        self.checks.is_instructor_initialized = True

    # ----------------------------------------------------------------
    # TOOL CALLING
    # ----------------------------------------------------------------

    def _build_tools(
        self, tools: List[ToolType], model: str
    ) -> List[Tool]:
        """Formats a list of tools for completion.

        Handles both function tools and string-based tools that need to be generated.
        String tools will be dynamically generated using the _generate_tool method.
        """
        formatted_tools = []

        if self.config.verbose:
            console.print(f"🔨 Building tools... Total tools: {len(tools)}")

        try:
            for tool in tools:
                if isinstance(tool, str):
                    if self.config.verbose:
                        console.print(f"🔨 Generating tool from name: {tool}")
                    generated_tool = _generate_tool(
                        client=self, tool_name=tool, model=model
                    )
                    formatted_tools.append(generated_tool)
                else:
                    if self.config.verbose:
                        console.print("🔨 Processing existing tool function")
                    formatted_tools.append(Tool(function=tool))

            for tool in formatted_tools:
                if not tool.name:
                    tool.name = function_calling.get_function_name(tool.function)
                if not tool.arguments:
                    tool.arguments = function_calling.get_function_arguments(
                        tool.function
                    )
                if not tool.formatted_function:
                    tool.formatted_function = function_calling.convert_to_openai_tool(
                        tool.function
                    )

        except Exception as e:
            raise ZyxError(
                f"❌ [bold dark_red]ToolConversionError[/bold dark_red] Failed to convert tool: {e}"
            )

        return formatted_tools

    def _execute_tools(
        self,
        tools: List[Tool],
        arguments: CompletionArguments,
        response: Completion,
    ) -> CompletionArguments:
        """Executes tools and returns updated arguments."""
        arguments.messages.append(response.choices[0].message.model_dump())

        if not response.choices[0].message.tool_calls:
            if self.config.verbose:
                console.print("🔨 No tool calls detected, skipping tool execution.")
            return arguments

        try:
            for tool_call in response.choices[0].message.tool_calls:
                name = tool_call.function.name
                for tool in tools:
                    if tool.name == name:
                        if self.config.verbose:
                            console.print(f"🔨 Executing tool: {name}")
                        args = json.loads(tool_call.function.arguments)
                        tool_output = tool._execute(verbose=self.config.verbose, **args)
                        arguments.messages = arguments.add_tool_execution_output(
                            arguments.messages, tool_call.id, tool_output
                        )
        except Exception as e:
            raise ZyxError(
                f"❌ [bold dark_red]ToolExecutionError[/bold dark_red] Failed to execute tool: {e}"
            )

        return arguments

    # ----------------------------------------------------------------
    # INSTRUCTOR (RESPONSE MODEL) HELPER
    # ----------------------------------------------------------------

    def _convert_to_response_model(
        self, response_model: Union[str, Type[BaseModel], Dict[str, Any]]
    ) -> BaseModel:
        """Converts a response model to a Pydantic model."""
        if isinstance(response_model, BaseModel):
            if self.config.verbose:
                console.print("🔨 Response model is already a Pydantic model.")
            return response_model

        if self.config.verbose:
            console.print(
                f"🔨 Converting output format of type: {type(response_model)} to Pydantic model."
            )

        if isinstance(response_model, str):
            return create_model("Response", __root__=(str, response_model))

        if isinstance(response_model, dict):
            return create_model("Response", **response_model)

    def _simplify_messages(
        self, arguments: CompletionArguments
    ) -> List[Dict[str, Any]]:
        """Formats messages for instructor."""
        formatted_messages = []
        ids = []

        for message in arguments.messages:
            if "role" in message and "content" in message:
                if message["role"] in ["user", "assistant"]:
                    if isinstance(message["content"], str) and message["content"]:
                        formatted_messages.append(
                            {"role": message["role"], "content": message["content"]}
                        )

            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    ids.append(tool_call["id"])
                formatted_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"*To complete your request, I ran the following tool call:\n\n"
                            f"Tool Name: {message['tool_calls'][0]['function']['name']}\n"
                            f"Arguments : {json.dumps(message['tool_calls'][0]['function']['arguments'])}\n\n"
                        ),
                    }
                )

        for message in formatted_messages:
            if "*" in message["content"]:
                for id in ids:
                    for tool_message in arguments.messages:
                        if (
                            tool_message["role"] == "tool"
                            and tool_message["tool_call_id"] == id
                        ):
                            message["content"] += (
                                f"\n\n*The output of the tool call was: {tool_message['content']}"
                            )

        return formatted_messages

    # ----------------------------------------------------------------
    # COMPLETION METHODS
    # ----------------------------------------------------------------

    def _chat_completion(self, *args, **kwargs) -> ChatCompletion:
        if self.config.verbose:
            console.print("⏳ Creating chat completion...")

        if self.config.provider == "litellm":
            import litellm

            litellm.drop_params = True

        try:
            return self._base_client.chat.completions.create(*args, **kwargs)
        except Exception as e:
            raise ZyxError(
                f"❌ [bold dark_red]CompletionError[/bold dark_red] Failed to create completion: {e}"
            )

    def _instructor_completion(
        self, mode: InstructorMode = "tool_call", *args, **kwargs
    ) -> Type[BaseModel]:
        """Runs an instructor completion, returns a Pydantic model."""
        if not self.checks.is_instructor_initialized:
            self._patch_client(mode)

        if self.checks.is_instructor_initialized and "mode" in kwargs:
            if kwargs["mode"] != mode:
                self._patch_client(mode)

        response = None

        try:
            if "stream" in kwargs:
                return self.instructor.chat.completions.create_partial(*args, **kwargs)
            else:
                response = self.instructor.chat.completions.create(*args, **kwargs)
                if response is None:
                    raise ZyxError(
                        f"❌ [bold dark_red]InstructorCompletionError[/bold dark_red] Failed to create instructor completion."
                    )
                return response
        except Exception as e:
            raise ZyxError(
                f"❌ [bold dark_red]InstructorCompletionError[/bold dark_red] Failed to create instructor completion: {e}"
            )

    def _assign_response_model_type(
        self,
        response_model: Optional[
            Union[Type[str], Type[int], Type[float], Type[bool], Type[list]]
        ],
    ) -> Type[BaseModel]:
        """Assigns a response model type."""
        if response_model is None:
            return None

        if response_model is str:
            return StringResponse
        elif response_model is int:
            return IntResponse
        elif response_model is float:
            return FloatResponse
        elif response_model is bool:
            return BoolResponse
        elif response_model is list:
            if any(isinstance(i, dict) for i in response_model):
                class NestedListResponse(BaseModel):
                    response: List[BaseModel]

                return NestedListResponse
            return ListResponse
        else:
            return None

    # ----------------------------------------------------------------
    # PUBLIC
    # ----------------------------------------------------------------

    def completion(
        self,
        messages: Union[str, List[Message]],
        model: Union[str, ChatModel] = ZYX_DEFAULT_MODEL,
        mode: Optional[InstructorMode] = "tool_call",
        response_model: Optional[
            Union[
                str,
                Type[BaseModel],
                Dict[str, Any],
                Type[str],
                Type[int],
                Type[float],
                Type[bool],
                Type[list],
            ]
        ] = None,
        run_tools: Optional[bool] = True,
        tools: Optional[List[Union[str, ToolType]]] = None,
        process: Optional[Process] = None,
        chat: Optional[bool] = None,
        progress_bar: Optional[bool] = None,
        verbose: Optional[bool] = None,
        max_retries: Optional[int] = DEFAULT_MAX_RETRIES,
        audio: Optional[ChatCompletionAudioParam] = None,
        frequency_penalty: Optional[float] = None,
        function_call: Optional[FunctionCall] = None,
        functions: Optional[Iterable[Function]] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        modalities: Optional[List[ChatCompletionModality]] = None,
        n: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = False,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        seed: Optional[int] = None,
        service_tier: Optional[Literal["auto", "default"]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        store: Optional[bool] = None,
        stream: Optional[Literal[False]] | Literal[True] = None,
        stream_options: Optional[ChatCompletionStreamOptionsParam] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        top_logprobs: Optional[int] = None,
        user: Optional[str] = None,
    ) -> Completion:
        """Creates a chat completion.
        
        Examples:
        ```python
        z.completion("hi", model = "gpt-4o")
        ```

        ```python
        z.completion("hi", model = "gpt-4o", response_model = str)
        ```

        ```python
        z.completion("hi", model = "gpt-4o", response_model = List[str])
        ```

        Parameters:
            messages (Union[str, List[Message]]): The messages to send to the model.
            model (Union[str, ChatModel]): The model to use for the completion.
            mode (Optional[InstructorMode]): The mode to use for the completion.
            response_model (Optional[Union[str, Type[BaseModel], Dict[str, Any], Type[str], Type[int], Type[float], Type[bool], Type[list]]]): The response model to use for the completion.
            run_tools (Optional[bool]): Whether to run tools for the completion.
            tools (Optional[List[Union[str, ToolType]]]): The tools to use for the completion.
            process (Optional[Process]): The process to use for the completion.
            chat (Optional[bool]): Whether to use chat mode for the completion.
            progress_bar (Optional[bool]): Whether to show a progress bar for the completion.
            audio (Optional[ChatCompletionAudioParam]): The audio parameters to use for the completion.
            frequency_penalty (Optional[float]): The frequency penalty to use for the completion.
            function_call (Optional[FunctionCall]): The function call to use for the completion.
            functions (Optional[Iterable[Function]]): The functions to use for the completion.
            logit_bias (Optional[Dict[str, int]]): The logit bias to use for the completion.
            logprobs (Optional[bool]): Whether to return log probabilities for the completion.
            max_completion_tokens (Optional[int]): The maximum number of completion tokens to use for the completion.
            max_tokens (Optional[int]): The maximum number of tokens to use for the completion.
            metadata (Optional[Dict[str, str]]): The metadata to use for the completion.
            modalities (Optional[List[ChatCompletionModality]]): The modalities to use for the completion.
            n (Optional[int]): The number of completions to generate.
            parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls for the completion.
            presence_penalty (Optional[float]): The presence penalty to use for the completion.
            response_format (Optional[ResponseFormat]): The response format to use for the completion.
            seed (Optional[int]): The seed to use for the completion.
            service_tier (Optional[Literal["auto", "default"]]): The service tier to use for the completion.
            stop (Optional[Union[str, List[str]]]): The stop sequence to use for the completion.
            store (Optional[bool]): Whether to store the completion.
            stream (Optional[Literal[False]] | Literal[True]): Whether to stream the completion.
            stream_options (Optional[ChatCompletionStreamOptionsParam]): The stream options to use for the completion.
            temperature (Optional[float]): The temperature to use for the completion.
            top_p (Optional[float]): The top p value to use for the completion.
            tool_choice (Optional[ChatCompletionToolChoiceOptionParam]): The tool choice to use for the completion.
            top_logprobs (Optional[int]): The top log probabilities to use for the completion.
            user (Optional[str]): The user to use for the completion.

        Returns:
            Completion: The completion.
        """
        if response_model:
            if response_model in [str, int, float, bool, list]:
                response_model = self._assign_response_model_type(response_model)

        if max_retries is not None and max_retries != DEFAULT_MAX_RETRIES:
            self.config.max_retries = max_retries

        if model is None:
            model = ZYX_DEFAULT_MODEL

        rec = self._recommend(
            model=model, base_url=self.config.base_url, api_key=self.config.api_key
        )

        if (
            rec.provider != self.config.provider
            or rec.model != model
            or rec.base_url != self.config.base_url
        ):
            if self.config.verbose:
                console.print(
                    f"🔨 Invalid Config Detected. Changing provider to: {rec.provider}"
                )
            self.config.provider = rec.provider
            model = rec.model
            self.config.base_url = rec.base_url
            self._init_client()

        if chat is not None or process is not None:
            Yikes("Chat & Process args are not yet implemented. Oops..")

        tools_ran = False
        self.config.verbose = verbose if verbose is not None else self.config.verbose

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        for message in messages:
            if "role" not in message or "content" not in message:
                raise ZyxError("Invalid message format: 'role' and 'content' are required.")

        if response_model:
            if not (isinstance(response_model, type) and issubclass(response_model, BaseModel)):
                response_model = self._convert_to_response_model(response_model)

        arguments = CompletionArguments(
            messages=messages,
            model=model,
            run_tools=run_tools,
            tools=tools,
            mode=mode,
            response_model=response_model,
            process=process,
            chat=chat,
            progress_bar=progress_bar,
            function_call=function_call,
            functions=functions,
            audio=audio,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            metadata=metadata,
            modalities=modalities,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            store=store,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            tool_choice=tool_choice,
            top_logprobs=top_logprobs,
            user=user,
        )

        if not tools and not response_model:
            try:
                if progress_bar:
                    with Progress(console=console, expand=True) as progress:
                        task_id = progress.add_task(
                            description="Creating chat completion...", total=None
                        )
                        response = self._chat_completion(**arguments.model_dump_POST())
                        progress.remove_task(task_id)
                        return response
                else:
                    return self._chat_completion(**arguments.model_dump_POST())
            except Exception as e:
                raise ZyxError(
                    f"❌ [bold dark_red]CompletionError[/bold dark_red] Failed to create completion: {e}"
                )

        if tools is not None:
            arguments.stream = False
            _tools = self._build_tools(tools, model)
            arguments.add_tools(_tools)

            try:
                if progress_bar:
                    with Progress(console=console, expand=True) as progress:
                        task_id = progress.add_task(
                            description="Creating chat completion...", total=None
                        )
                        response = self._chat_completion(**arguments.model_dump_POST())
                        progress.remove_task(task_id)
                else:
                    response = self._chat_completion(**arguments.model_dump_POST())
            except Exception as e:
                console.print(f"❌ [bold dark_red]CompletionError[/bold dark_red]: {e}")
                raise

            if not run_tools and not response_model:
                return response

        if run_tools and tools is not None:
            arguments = self._execute_tools(_tools, arguments, response)
            tools_ran = True

        if response_model is None and tools_ran:
            try:
                if progress_bar:
                    with Progress(console=console, expand=True) as progress:
                        arguments.stream = stream
                        task_id = progress.add_task(
                            description="Executing tools...", total=None
                        )
                        response = self._chat_completion(**arguments.model_dump_POST())
                        progress.remove_task(task_id)

                        if response.choices[0].message.tool_calls:
                            arguments = self._execute_tools(_tools, arguments, response)
                            response = self._chat_completion(**arguments.model_dump_POST())
                            return response
                        else:
                            return response
                else:
                    arguments.stream = stream
                    response = self._chat_completion(**arguments.model_dump_POST())

                    if response.choices[0].message.tool_calls:
                        arguments = self._execute_tools(_tools, arguments, response)
                        return self._chat_completion(**arguments.model_dump_POST())
                    else:
                        return response
            except Exception as e:
                raise ZyxError(
                    f"❌ [bold dark_red]CompletionError[/bold dark_red] Failed to create completion: {e}"
                )

        if response_model is not None:
            if tools_ran:
                if self.config.verbose:
                    console.print("🔨 Running response model...")

                Yikes(
                    "Currently Instructor & Tool Implementation is not working well together, a hack is implemented to fix this."
                )
                Yikes(
                    "Running tools through instructor models will be implemented in the future."
                )

                arguments.messages = self._simplify_messages(arguments)
                arguments.messages = arguments.repair_messages(verbose=self.config.verbose)

            try:
                if progress_bar:
                    with Progress(console=console, expand=True) as progress:
                        task_id = progress.add_task(
                            description="Generating Instructor response...", total=None
                        )
                        response = self._instructor_completion(
                            mode=mode, **arguments.model_dump_POST(True, response_model)
                        )
                        progress.remove_task(task_id)
                else:
                    response = self._instructor_completion(
                        mode=mode, **arguments.model_dump_POST(True, response_model)
                    )

                # New logic for handling specific response model types
                if isinstance(response_model, type) and not issubclass(response_model, BaseModel):
                    if response_model is str:
                        return response.choices[0].message.content
                    else:
                        return response_model(response)

                return response

            except Exception as e:
                raise ZyxError(
                    f"❌ [bold dark_red]CompletionError[/bold dark_red] Failed to create completion: {e}"
                )


def completion(
    messages: Union[str, List[Message]],
    model: Union[str, ChatModel] = ZYX_DEFAULT_MODEL,
    mode: Optional[InstructorMode] = "tool_call",
    response_model: Optional[
        Union[
            str,
            Type[BaseModel],
            Dict[str, Any],
            Type[str],
            Type[int],
            Type[float],
            Type[bool],
            Type[list],
        ]
    ] = None,
    provider: Optional[Union[Literal["openai", "litellm"], ClientProvider]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = DEFAULT_TIMEOUT,
    max_retries: Optional[int] = DEFAULT_MAX_RETRIES,
    run_tools: Optional[bool] = True,
    tools: Optional[List[Union[str, ToolType]]] = None,
    process: Optional[Process] = None,
    chat: Optional[bool] = None,
    progress_bar: Optional[bool] = None,
    audio: Optional[ChatCompletionAudioParam] = None,
    frequency_penalty: Optional[float] = None,
    function_call: Optional[FunctionCall] = None,
    functions: Optional[Iterable[Function]] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    logprobs: Optional[bool] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
    modalities: Optional[List[ChatCompletionModality]] = None,
    n: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = False,
    presence_penalty: Optional[float] = None,
    response_format: Optional[ResponseFormat] = None,
    seed: Optional[int] = None,
    service_tier: Optional[Literal["auto", "default"]] = None,
    stop: Optional[Union[str, List[str]]] = None,
    store: Optional[bool] = None,
    stream: Optional[Literal[False]] | Literal[True] = None,
    stream_options: Optional[ChatCompletionStreamOptionsParam] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
    top_logprobs: Optional[int] = None,
    user: Optional[str] = None,
    default_headers: Optional[Mapping[str, str]] = None,
    default_query: Optional[Mapping[str, object]] = None,
    verify_ssl: Optional[bool] = None,
    http_args: Optional[Mapping[str, object]] = None,
    http_client: Optional[httpx.Client] = None,
    verbose: Optional[bool] = None,
) -> Completion:
    """Creates a chat completion."""
    if provider is None:
        rec = Client._recommend(model=model, base_url=base_url, api_key=api_key)
    else:
        rec = Client._recommend(model=model, base_url=base_url, api_key=api_key, provider=provider)

    client = Client(
        provider=rec.provider,
        api_key=rec.api_key,
        base_url=rec.base_url,
        organization=organization,
        project=project,
        timeout=timeout,
        max_retries=max_retries,
        default_headers=default_headers,
        default_query=default_query,
        verify_ssl=verify_ssl,
        http_args=http_args,
        http_client=http_client,
        verbose=verbose,
    )

    return client.completion(
        messages=messages,
        model=rec.model,
        mode=mode,
        response_model=response_model,
        run_tools=run_tools,
        tools=tools,
        process=process,
        chat=chat,
        progress_bar=progress_bar,
        audio=audio,
        frequency_penalty=frequency_penalty,
        function_call=function_call,
        functions=functions,
        logit_bias=logit_bias,
        logprobs=logprobs,
        max_completion_tokens=max_completion_tokens,
        max_tokens=max_tokens,
        metadata=metadata,
        modalities=modalities,
        n=n,
        parallel_tool_calls=parallel_tool_calls,
        presence_penalty=presence_penalty,
        response_format=response_format,
        seed=seed,
        service_tier=service_tier,
        stop=stop,
        store=store,
        stream=stream,
        stream_options=stream_options,
        temperature=temperature,
        top_p=top_p,
        tool_choice=tool_choice,
        top_logprobs=top_logprobs,
        user=user,
    )


if __name__ == "__main__":
    print(completion("hi", model = "anthropic/claude-3-5-sonnet-latest", progress_bar=True))
