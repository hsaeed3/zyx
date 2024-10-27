# xnano._base
# base llm client
# handles all llm completions

# NOTE
# you do not want to be here....
# dont use this directly (unless you want to idk)

from __future__ import annotations

__all__ = [
    "Client",
    "completion"
]

# core utils
from ._rich import console, logger

# exceptions
from ._exceptions import ZyxError, Yikes

# coder
from .resources.utils.tool_generator import _generate_tool

# TYPES
from .resources.types import completion_create_params as params
from .resources.types.config import client as configtypes

# Utils
# tool calls
from .resources.utils import function_calling

# model outputs
from .resources.types import model_outputs as outputs

import json
import httpx
import openai
from pydantic import BaseModel, create_model
from typing import Any, List, Literal, Mapping, Optional, Type, Union, Dict, Iterable
from rich.progress import Progress


# LiteLLM Generic
LiteLLM = Type["LiteLLM"]


# INTERNAL RECOMMENDATION CLASS
class Recommendation(BaseModel):
    model : Optional[str] = None
    provider : Optional[str] = None
    base_url : Optional[str] = None
    api_key : Optional[str] = None


# client
class Client:

    """Base LLM Completions Client for Zyx."""

    def __init__(
            self,
            # PROVIDER // instantiates between openai.OpenAI or litellm.LiteLLM
            provider : Union[Literal["openai", "litellm"], configtypes.ClientProvider] = "openai",
            # base client args
            api_key : Optional[str] = None,
            base_url : Optional[str] = None,
            organization : Optional[str] = None,
            project : Optional[str] = None,
            timeout : Optional[Union[float, httpx.Timeout]] = configtypes.DEFAULT_TIMEOUT,
            max_retries : Optional[int] = configtypes.DEFAULT_MAX_RETRIES,
            default_headers : Optional[Mapping[str, str]] = None,
            default_query : Optional[Mapping[str, object]] = None,
            # pruned httpx args
            verify_ssl : Optional[bool] = None,
            http_args : Optional[Mapping[str, object]] = None,
            # httpx client // only openai supports this
            http_client : Optional[httpx.Client] = None,
            # verbosity
            verbose : Optional[bool] = None,
    ):

        # init client config
        self.config = configtypes.ClientConfig(
            provider=provider,
            api_key=api_key, base_url=base_url, organization=organization, project=project,
            timeout=timeout, max_retries=max_retries, default_headers=default_headers, default_query=default_query,
            verify_ssl=verify_ssl, http_args=http_args, http_client=http_client, verbose=verbose
        )

        # init client checks
        self.checks = configtypes.ClientChecks()

        # init client
        self._init_client()


    # ----------------------------------------------------------------
    # RECOMMENDATION
    # ----------------------------------------------------------------
    @staticmethod
    def _recommend(model : str, base_url : Optional[str] = None, api_key : Optional[str] = None) -> Recommendation:
        """Recommends args based on the given model name."""

        if base_url:
            return Recommendation(provider = "openai",model = model, base_url = base_url, api_key = api_key)

        rec = Recommendation()

        if model.startswith(
            ("gpt-", "o1", "openai/", "chatgpt-", "ollama/")
        ):
            rec.provider = "openai"

            if model.startswith(
                ("openai/", "chatgpt-", "o1", "gpt-")
            ):
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

    # openai client
    def __get_openai_client(self) -> openai.OpenAI:
        """Returns the OpenAI client."""

        from openai import OpenAI

        try:
            return OpenAI(
                api_key=self.config.api_key, base_url=self.config.base_url, organization=self.config.organization, project=self.config.project,
                timeout=self.config.timeout, max_retries=self.config.max_retries, default_headers=self.config.default_headers, default_query=self.config.default_query,
                http_client=self.config.http_client
            )
        except Exception as e:
            raise ZyxError(f"❌ [bold dark_red]ClientInitializationError[/bold dark_red] Failed to initialize [bold]OpenAI[/bold] client: {e}")

    # litellm client
    def __get_litellm_client(self) -> LiteLLM:
        """Returns the LiteLLM client."""

        from litellm import LiteLLM

        try:
            return LiteLLM(
                api_key=self.config.api_key, base_url=self.config.base_url, organization=self.config.organization,
                timeout=self.config.timeout, max_retries=self.config.max_retries, default_headers=self.config.default_headers
            )
        except Exception as e:
            raise ZyxError(f"❌ [bold dark_red]ClientInitializationError[/bold dark_red] Failed to initialize [bold]LiteLLM[/bold] client: {e}")

    # init client
    def _init_client(self) -> Union[openai.OpenAI, LiteLLM]:
        """Initializes the client based on the provider."""

        if self.config.provider == "openai":

            # NOTE -- verify ssl & http args
            # added to make it easier to pass in httpx args
            if self.config.verify_ssl is not None or self.config.http_args is not None:
                if self.config.http_client is not None:
                    logger.warning(
                        "⚠️ [bold]Verify SSL[/bold] or [bold]HTTP args[/bold] detected, any passed http client will be ignored."
                )
                self.config.http_client = httpx.Client(
                    verify=self.config.verify_ssl, **self.config.http_args
                )

            # init openai client
            self._base_client = self.__get_openai_client()

        else:

            # init litellm client
            self._base_client = self.__get_litellm_client()

        self.checks.is_client_initialized = True

        if self.config.verbose:
            console.print(f"✅ Initialized {self.config.provider} client.")

    # ----------------------------------------------------------------
    # INSTRUCTOR PATCHES
    # ----------------------------------------------------------------

    # openai client
    def __instructor_patch_openai_client(
            self,
            mode : params.InstructorMode = "tool_call"
    ) -> openai.OpenAI:
        """Patches the OpenAI client with instructor & given mode"""

        from instructor import from_openai, Mode

        try:
            return from_openai(self._base_client, mode=Mode(mode))
        except Exception as e:
            raise ZyxError(f"❌ [bold dark_red]InstructorPatchError[/bold dark_red] Failed to patch [bold]OpenAI[/bold] client: {e}")

    # litellm client
    def __instructor_patch_litellm_client(
            self,
            mode : params.InstructorMode = "tool_call"
    ) -> LiteLLM:
        """Patches the LiteLLM client with instructor."""

        from instructor import patch as litellm_patch, Mode

        try:
            return litellm_patch(self._base_client, mode=Mode(mode))
        except Exception as e:
            raise ZyxError(f"❌ [bold dark_red]InstructorPatchError[/bold dark_red] Failed to patch [bold]LiteLLM[/bold] client: {e}")

    # patch client
    def _patch_client(self, mode : params.InstructorMode = "tool_call") -> Union[openai.OpenAI, LiteLLM]:
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

    # tool builder
    def _build_tools(
            self,
            tools: List[params.ToolType],
            model: str
        ) -> List[params.Tool]:
        """Formats a list of tools for completion.

        Handles both function tools and string-based tools that need to be generated.
        String tools will be dynamically generated using the _generate_tool method.
        """

        formatted_tools = []

        if self.config.verbose:
            console.print(f"🔨 Building tools... Total tools: {len(tools)}")

        try:
            for tool in tools:
                # Handle string-based tool generation
                if isinstance(tool, str):
                    if self.config.verbose:
                        console.print(f"🔨 Generating tool from name: {tool}")
                    generated_tool = _generate_tool(
                        client=self,
                        tool_name=tool,
                        model=model
                    )
                    formatted_tools.append(generated_tool)
                # Handle existing function tools
                else:
                    if self.config.verbose:
                        console.print(f"🔨 Processing existing tool function")
                    formatted_tools.append(
                        params.Tool(function=tool)
                    )

            # Process all tools to ensure proper formatting
            for tool in formatted_tools:
                if not tool.name:
                    tool.name = function_calling.get_function_name(tool.function)
                if not tool.arguments:
                    tool.arguments = function_calling.get_function_arguments(tool.function)
                if not tool.formatted_function:
                    tool.formatted_function = function_calling.convert_to_openai_tool(tool.function)

        except Exception as e:
            raise ZyxError(f"❌ [bold dark_red]ToolConversionError[/bold dark_red] Failed to convert tool: {e}")

        return formatted_tools


    # tool execution handler
    def _execute_tools(
            self,
            tools: List[params.Tool],
            arguments: params.CompletionArguments,
            response: params.Completion
        ) -> params.CompletionArguments:
        """Executes tools and returns a list of messages in args."""

        # append response to arg messages
        arguments.messages.append(
            response.choices[0].message.model_dump()
        )

        # no tool calls
        if not response.choices[0].message.tool_calls:
            if self.config.verbose:
                console.print("🔨 No tool calls detected, skipping tool execution.")
            return arguments

        # execute tools
        try:
            for tool_call in response.choices[0].message.tool_calls:

                # get call name
                name = tool_call.function.name

                # match name to tool
                for tool in tools:
                    if tool.name == name:

                        if self.config.verbose:
                            console.print(f"🔨 Executing tool: {name}")

                        # collect args from call
                        args = json.loads(tool_call.function.arguments)

                        # execute tool and get output
                        tool_output = tool._execute(verbose=self.config.verbose, **args)

                        # add tool response to messages using the method
                        arguments.messages = arguments.add_tool_execution_output(arguments.messages, tool_call.id, tool_output)

        except Exception as e:
            raise ZyxError(f"❌ [bold dark_red]ToolExecutionError[/bold dark_red] Failed to execute tool: {e}")

        return arguments

    # ----------------------------------------------------------------
    # INSTRUCTOR (RESPONSE MODEL) HELPER
    # ----------------------------------------------------------------

    # convert to response model
    def _convert_to_response_model(
            self,
            response_model : Union[str, Type[BaseModel], dict[str, Any]]
    ) -> BaseModel:
        """Converts a response model to a Pydantic model."""

        # if all is well :)
        if isinstance(response_model, BaseModel):

            if self.config.verbose:
                console.print("🔨 Response model is already a Pydantic model.")

            return response_model

        if self.config.verbose:
            console.print(f"🔨 Converting output format of type: {type(response_model)} to Pydantic model.")

        # if not all is well?
        if isinstance(response_model, str):
            return create_model("Response", __root__=(str, response_model))

        if isinstance(response_model, dict):
            return create_model("Response", __root__=response_model)

    # instructor message formatter
    def _simplify_messages(
            self,
            arguments : params.CompletionArguments
    ) -> List[Dict[str, Any]]:
        """Formats messages for instructor."""

        formatted_messages = []
        ids = []

        for message in arguments.messages:

            # if message is a standard user/assistant message
            if "role" in message and "content" in message:
                if message["role"] in ["user", "assistant"]:
                    # ensure content is a string and not empty
                    if isinstance(message["content"], str) and message["content"]:
                        formatted_messages.append(
                            {"role": message["role"], "content": message["content"]}
                        )

            # if the message is a tool call -> with corrosponding (if any) "role" : "tool" messages
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
                        )
                    }
                )

        for message in formatted_messages:
            # check for * to identify tool call messages
            # and append the tool output to message content with corresponding tool call id
            if "*" in message["content"]:

                # match id
                for id in ids:

                    # ensure tool message
                    for tool_message in arguments.messages:

                        # append if id match
                        if tool_message["role"] == "tool" and tool_message["tool_call_id"] == id:
                            message["content"] += f"\n\n*The output of the tool call was: {tool_message['content']}"

        return formatted_messages


    # ----------------------------------------------------------------
    # COMPLETION METHODS
    # ----------------------------------------------------------------

    # non response model completion
    def _chat_completion(
            self,
            *args,
            **kwargs
    ) -> params.ChatCompletion:

        if self.config.verbose:
            console.print("⏳ Creating chat completion...")

        # run litellm.drop_params
        # ensures all args are valid for the client
        if self.config.provider == "litellm":

            import litellm
            litellm.drop_params = True


        try:
            return self._base_client.chat.completions.create(*args, **kwargs)


        except Exception as e:
            raise ZyxError(f"❌ [bold dark_red]CompletionError[/bold dark_red] Failed to create completion: {e}")

    # instructor completion
    def _instructor_completion(
            self,
            mode : params.InstructorMode = "tool_call",
            *args,
            **kwargs
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
                    raise ZyxError(f"❌ [bold dark_red]InstructorCompletionError[/bold dark_red] Failed to create instructor completion.")

                return response

        except Exception as e:
            raise ZyxError(f"❌ [bold dark_red]InstructorCompletionError[/bold dark_red] Failed to create instructor completion: {e}")


    def _assign_response_model_type(
            self,
            response_model : Optional[Union[Type[str], Type[int], Type[float], Type[bool], Type[list]]]
    ) -> Type[BaseModel]:
        """Assigns a response model type."""

        if response_model is None:
            return None

        if response_model is str:
            return outputs.StringResponse

        elif response_model is int:
            return outputs.IntResponse

        elif response_model is float:
            return outputs.FloatResponse

        elif response_model is bool:
            return outputs.BoolResponse

        elif response_model is list:
            if any(isinstance(i, dict) for i in response_model):
                class NestedListResponse(BaseModel):
                    response: List[BaseModel]
                return NestedListResponse
            return outputs.ListResponse

        else:
            return None

    # ----------------------------------------------------------------
    # PUBLIC
    # ----------------------------------------------------------------

    def completion(
            self,
            # NOTE // REQUIRED
            messages : Union[str, List[params.Message]],
            # required but handled (default is gpt-4o-mini // use zyx.change_default_model to change)
            model : Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
            # INSTRUCTOR
            mode : Optional[params.InstructorMode] = "tool_call",
            response_model : Optional[
                Union[str, Type[BaseModel], dict[str, Any], Type[str], Type[int], Type[float], Type[bool], Type[list]]
            ] = None,
            # TOOL CALLING
            run_tools : Optional[bool] = True,
            tools: Optional[List[Union[str, params.ToolType]]] = None,
            # OTHER ZYX ARGS
            process : Optional[params.Process] = None,
            # NOT IMPLEMENTED YET
            chat : Optional[bool] = None,
            progress_bar : Optional[bool] = None,
            # VERBOSITY
            verbose : Optional[bool] = None,
            # ARGS

            # max retries (instructor)
            max_retries : Optional[int] = params.DEFAULT_MAX_RETRIES,

            audio : Optional[params.ChatCompletionAudioParam] = None,
            frequency_penalty : Optional[float] = None,
            function_call : Optional[params.FunctionCall] = None,
            functions : Optional[Iterable[params.Function]] = None,
            logit_bias : Optional[Dict[str, int]] = None,
            logprobs : Optional[bool] = None,
            max_completion_tokens : Optional[int] = None,
            max_tokens : Optional[int] = None,
            metadata : Optional[Dict[str, str]] = None,
            modalities : Optional[List[params.ChatCompletionModality]] = None,
            n : Optional[int] = None,
            parallel_tool_calls : Optional[bool] = False,
            presence_penalty : Optional[float] = None,
            response_format : Optional[params.ResponseFormat] = None,
            seed : Optional[int] = None,
            service_tier : Optional[Literal["auto", "default"]] = None,
            stop : Optional[Union[str, List[str]]] = None,
            store : Optional[bool] = None,
            stream : Optional[Literal[False]] | Literal[True] = None,
            stream_options : Optional[params.ChatCompletionStreamOptionsParam] = None,
            temperature : Optional[float] = None,
            top_p : Optional[float] = None,
            tool_choice : Optional[params.ChatCompletionToolChoiceOptionParam] = None,
            top_logprobs : Optional[int] = None,
            user : Optional[str] = None,
    ) -> params.Completion:
        """Creates a chat completion."""

        if response_model:
            if response_model in [str, int, float, bool, list]:
                response_model = self._assign_response_model_type(response_model)

        if max_retries is not None and max_retries != params.DEFAULT_MAX_RETRIES:
            self.config.max_retries = max_retries

        if model is None:
            model = params.ZYX_DEFAULT_MODEL

        rec = self._recommend(model = model, base_url = self.config.base_url, api_key = self.config.api_key)

        if rec.provider != self.config.provider or rec.model != model or rec.base_url != self.config.base_url:

            if self.config.verbose:
                console.print(f"🔨 Invalid Config Detected. Changing provider to: {rec.provider}")

            self.config.provider = rec.provider

            model = rec.model

            self.config.base_url = rec.base_url

            self._init_client()

        if chat is not None or process is not None:
            Yikes(f"Chat & Process args are not yet implemented. Oops..")

        # placeholder for tool exec
        tools_ran = False

        # set verbosity
        self.config.verbose = verbose if verbose is not None else self.config.verbose

        # Ensure messages are correctly formatted
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Check for invalid message structure
        for message in messages:
            if "role" not in message or "content" not in message:
                raise ZyxError("Invalid message format: 'role' and 'content' are required.")

        # convert response model
        if response_model:
            if not (isinstance(response_model, type) and issubclass(response_model, BaseModel)):
                response_model = self._convert_to_response_model(response_model)

        # init args
        arguments = params.CompletionArguments(
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
                    with Progress(
                        console = console, expand = True,
                    ) as progress:
                        task_id = progress.add_task(
                            description = "Creating chat completion...",
                            total = None
                        )
                        response = self._chat_completion(**arguments.model_dump_POST())
                        # delete loader
                        progress.remove_task(task_id)
                        return response
                else:
                    return self._chat_completion(**arguments.model_dump_POST())
            except Exception as e:
                raise ZyxError(f"❌ [bold dark_red]CompletionError[/bold dark_red] Failed to create completion: {e}")

        # tool handling
        if tools is not None:

            arguments.stream = False

            # Build tools and add to arguments
            _tools = self._build_tools(tools, model)
            arguments.add_tools(_tools)

            # Run completion
            try:
                if progress_bar:
                    with Progress(
                        console = console, expand = True,
                    ) as progress:
                        task_id = progress.add_task(
                            description = "Creating chat completion...",
                            total = None
                        )
                        response = self._chat_completion(**arguments.model_dump_POST())
                        progress.remove_task(task_id)
                else:
                    response = self._chat_completion(**arguments.model_dump_POST())
            except Exception as e:
                console.print(f"❌ [bold dark_red]CompletionError[/bold dark_red]: {e}")
                raise

            # Check for tool calls in the response
            if not run_tools and not response_model:
                return response

        if run_tools and tools is not None:
            # Execute tools
            arguments = self._execute_tools(_tools, arguments, response)

            tools_ran = True

        if response_model is None and tools_ran:
            try:
                if progress_bar:
                    with Progress(
                        console = console, expand = True,
                    ) as progress:
                        arguments.stream = stream
                        task_id = progress.add_task(
                            description = "Executing tools...",
                            total = None
                        )
                        response = self._chat_completion(**arguments.model_dump_POST())
                        progress.remove_task(task_id)

                        if response.choices[0].message.tool_calls:

                            # execute tools
                            arguments = self._execute_tools(_tools, arguments, response)

                            # generate one last time
                            response = self._chat_completion(**arguments.model_dump_POST())

                            return response

                        else:
                            return response
                else:
                    arguments.stream = stream

                    response = self._chat_completion(**arguments.model_dump_POST())

                    if response.choices[0].message.tool_calls:

                        # execute tools
                        arguments = self._execute_tools(_tools, arguments, response)

                        # generate one last time
                        return self._chat_completion(**arguments.model_dump_POST())

                    else:
                        return response
            except Exception as e:
                raise ZyxError(f"❌ [bold dark_red]CompletionError[/bold dark_red] Failed to create completion: {e}")

        # TODO // IMPORTANT // -----------------------------------
        # instructor has not been handling
        # tool message threads very well.
        # a hack is implemented
        # --------------------------------------------------------
        if response_model is not None:
            # append response if tools ran
            if tools_ran:

                if self.config.verbose:
                    console.print("🔨 Running response model...")

                Yikes(f"Currently Instructor & Tool Implementation is not working well together, a hack is implemented to fix this.")
                Yikes(f"Running tools through instructor models will be implemented in the future.")


                arguments.messages = self._simplify_messages(arguments)

                arguments.messages = arguments.repair_messages(verbose = self.config.verbose)

            # instructor completion
            try:
                if progress_bar:
                    with Progress(
                        console = console, expand = True,
                    ) as progress:
                        task_id = progress.add_task(
                            description = "Generating Instructor response...",
                            total = None
                        )
                        response = self._instructor_completion(mode = mode, **arguments.model_dump_POST(True, response_model))
                        progress.remove_task(task_id)
                        return response
                else:
                    return self._instructor_completion(mode = mode, **arguments.model_dump_POST(True, response_model))
            except Exception as e:
                raise ZyxError(f"❌ [bold dark_red]CompletionError[/bold dark_red] Failed to create completion: {e}")


# -- completion --

def completion(
        messages : Union[str, List[params.Message]],
        model : Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        mode : Optional[params.InstructorMode] = "tool_call",
        response_model : Optional[
            Union[str, Type[BaseModel], dict[str, Any], Type[str], Type[int], Type[float], Type[bool], Type[list]]
        ] = None,
        # PROVIDER // instantiates between openai.OpenAI or litellm.LiteLLM
        provider : Optional[Union[Literal["openai", "litellm"], params.ClientProvider]] = None,
        # base client args
        api_key : Optional[str] = None,
        base_url : Optional[str] = None,
        organization : Optional[str] = None,
        project : Optional[str] = None,
        timeout : Optional[Union[float, httpx.Timeout]] = params.DEFAULT_TIMEOUT,
        max_retries : Optional[int] = params.DEFAULT_MAX_RETRIES,
        # TOOL CALLING
        run_tools : Optional[bool] = True,
        tools: Optional[List[Union[str, params.ToolType]]] = None,
        # OTHER ZYX ARGS
        process : Optional[params.Process] = None,
        chat : Optional[bool] = None,
        progress_bar : Optional[bool] = None,
        # ARGS
        audio : Optional[params.ChatCompletionAudioParam] = None,
        frequency_penalty : Optional[float] = None,
        function_call : Optional[params.FunctionCall] = None,
        functions : Optional[Iterable[params.Function]] = None,
        logit_bias : Optional[Dict[str, int]] = None,
        logprobs : Optional[bool] = None,
        max_completion_tokens : Optional[int] = None,
        max_tokens : Optional[int] = None,
        metadata : Optional[Dict[str, str]] = None,
        modalities : Optional[List[params.ChatCompletionModality]] = None,
        n : Optional[int] = None,
        parallel_tool_calls : Optional[bool] = False,
        presence_penalty : Optional[float] = None,
        response_format : Optional[params.ResponseFormat] = None,
        seed : Optional[int] = None,
        service_tier : Optional[Literal["auto", "default"]] = None,
        stop : Optional[Union[str, List[str]]] = None,
        store : Optional[bool] = None,
        stream : Optional[Literal[False]] | Literal[True] = None,
        stream_options : Optional[params.ChatCompletionStreamOptionsParam] = None,
        temperature : Optional[float] = None,
        top_p : Optional[float] = None,
        tool_choice : Optional[params.ChatCompletionToolChoiceOptionParam] = None,
        top_logprobs : Optional[int] = None,
        user : Optional[str] = None,
        default_headers : Optional[Mapping[str, str]] = None,
        default_query : Optional[Mapping[str, object]] = None,
        # pruned httpx args
        verify_ssl : Optional[bool] = None,
        http_args : Optional[Mapping[str, object]] = None,
        # httpx client // only openai supports this
        http_client : Optional[httpx.Client] = None,
        # verbosity
        verbose : Optional[bool] = None,

) -> params.Completion:
    """Creates a chat completion."""

    if provider is None:
        rec = Client._recommend(model = model, base_url = base_url, api_key = api_key)
    else:
        rec = Client._recommend(model = model, base_url = base_url, api_key = api_key, provider = provider)

    client = Client(
        provider = rec.provider, api_key = rec.api_key, base_url = rec.base_url,
        organization = organization, project = project, timeout = timeout,
        max_retries = max_retries, default_headers = default_headers,
        default_query = default_query, verify_ssl = verify_ssl, http_args = http_args,
        http_client = http_client, verbose = verbose
    )

    return client.completion(
        messages = messages, model = rec.model, mode = mode, response_model = response_model,
        run_tools = run_tools, tools = tools, process = process, chat = chat,
        progress_bar = progress_bar, audio = audio, frequency_penalty = frequency_penalty,
        function_call = function_call, functions = functions, logit_bias = logit_bias,
        logprobs = logprobs, max_completion_tokens = max_completion_tokens, max_tokens = max_tokens,
        metadata = metadata, modalities = modalities, n = n, parallel_tool_calls = parallel_tool_calls,
        presence_penalty = presence_penalty, response_format = response_format, seed = seed,
        service_tier = service_tier, stop = stop, store = store, stream = stream,
        stream_options = stream_options, temperature = temperature, top_p = top_p,
        tool_choice = tool_choice, top_logprobs = top_logprobs, user = user
    )


if __name__ == "__main__":

    print(
        completion(
            "hi"
        )
    )
