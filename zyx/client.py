from .lib.utils.logger import get_logger
from .lib.utils.convert_to_openai_tool import convert_to_openai_tool
from instructor import Mode

logger = get_logger(__name__)

import enum
import json
from pydantic import BaseModel
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)


## -- Instructor Configuration -- ##


InstructorMode = Literal[
    "function_call",
    "parallel_tool_call",
    "tool_call",
    "mistral_tools",
    "json_mode",
    "json_o1",
    "markdown_json_mode",
    "json_schema_mode",
    "anthropic_tools",
    "anthropic_json",
    "cohere_tools",
    "vertexai_tools",
    "vertexai_json",
    "gemini_json",
    "gemini_tools",
    "json_object",
    "tools_strict",
]


def get_mode(mode: InstructorMode) -> Mode:
    return Mode(mode)


## -- Tool Handling -- ##


ToolType = Union[Dict[str, Any], Type[BaseModel], Callable]


class Tool(BaseModel):
    name: Optional[str]
    tool: ToolType
    openai_tool: Optional[Dict[str, Any]]


class ToolResponse(BaseModel):
    name: Any
    args: Dict[str, Any]
    output: Optional[Any]


## -- Client Configuration Models -- ##


class ClientConfig(BaseModel):
    api_key: Optional[str]
    base_url: Optional[str]
    organization: Optional[str]
    verbose: bool


class ClientProviders(BaseModel):
    client: Optional[Any] = None
    instructor: Optional[Any] = None


## -- Completion Helpers -- ##


class CompletionArgs(BaseModel):
    messages: List[Dict[str, str]]
    model: str
    response_model: Optional[Type[BaseModel]]
    tools: Optional[List[Dict[str, Any]]]
    parallel_tool_calls: Optional[bool]
    tool_choice: Optional[Literal["none", "auto", "required"]]
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    stop: Optional[List[str]]
    stream: Optional[bool]


ChatCompletion = Type["ChatCompletion"]
ModelResponse = Type["ModelResponse"]


CompletionResponse = Union[Type[BaseModel], ChatCompletion, ModelResponse, Generator]


# ------------------ #
#
# Client
#
# ------------------ #


class Client:
    """
    Base class for all LLM completions in the zyx library.
    Runs using either the OpenAI or LiteLLM client libraries.
    Uses Instructor to patch the LLM client for tool calling & structured outputs.
    """

    @staticmethod
    def recommend_client_by_model(
        model: str, base_url: Optional[str] = None, api_key: Optional[str] = None
    ) -> tuple[
        Literal["openai", "litellm"], Optional[str], Optional[str], Optional[str]
    ]:
        """Recommends the client to use for the given model. Used in one-shot completions.

        Args:
            model: str: The model to recommend the client for.
            base_url: Optional[str]: The base URL to use for the client.
            api_key: Optional[str]: The API key to use for the client.

        Returns:
            tuple[Literal["openai", "litellm"], Optional[str], Optional[str], Optional[str]]: The recommended client, model, base URL, and API key.
        """
        if base_url is not None:
            client = "openai"

            return client, model, base_url, api_key

        if model.startswith(("gpt-", "o1", "openai/")):
            if model.startswith("openai/"):
                model = model[7:]

            client = "openai"

            return client, model, base_url, api_key

        elif model.startswith("ollama/"):
            model = model[7:]

            client = "openai"
            if not base_url:
                base_url = "http://localhost:11434/v1"

            if not api_key:
                api_key = "ollama"

            return client, model, base_url, api_key

        else:
            client = "litellm"

            return client, model, base_url, api_key

    @staticmethod
    def format_to_openai_tools(tools: List[ToolType]) -> List[Tool]:
        """Converts the tools to a list of dictionaries.

        Args:
            tools: List[ToolType]: The tools to convert.

        Returns:
            List[Tool]: The converted tools.
        """

        formatted_tools = []
        for tool in tools:
            formatted_tool = convert_to_openai_tool(tool)
            formatted_tools.append(
                Tool(
                    name=formatted_tool["function"]["name"],
                    tool=tool,
                    openai_tool=formatted_tool,
                )
            )

        return formatted_tools

    @staticmethod
    def get_tool_dict(tools: List[Tool]) -> List[Dict[str, Any]]:
        """Converts the tools to a list of dictionaries.

        Args:
            tools: List[Tool]: The tools to convert.

        Returns:
            List[Dict[str, Any]]: The converted tools.
        """

        tool_dict = []
        for tool in tools:
            tool_dict.append(tool.openai_tool)

        return tool_dict

    @staticmethod
    def format_messages(
        messages: Union[str, list[dict]] = None,
        verbose: Optional[bool] = False,
        type: Optional[Literal["user", "system", "assistant"]] = "user",
    ) -> list[dict]:
        """Formats the messages into a list of dictionaries.

        Args:
            messages: Union[str, list[dict]]: The messages to format.
            verbose: bool: Whether to log the formatting process.

        Returns:
            list[dict]: The formatted messages.
        """

        try:
            if isinstance(messages, str):
                if verbose:
                    print(f"Converting string to message format.")

                return [{"role": type, "content": messages}]
            elif isinstance(messages, list) and all(
                isinstance(m, dict) for m in messages
            ):
                if verbose:
                    print(f"Messages are in the correct format.")

                return messages
            else:
                raise ValueError("Invalid message format")
        except Exception as e:
            print(f"Error formatting messages: {e}")
            return []

    @staticmethod
    def does_system_prompt_exist(messages: list[dict]) -> bool:
        """Simple boolean check to see if a system prompt exists in the messages.

        Args:
            messages: list[dict]: The messages to check.

        Returns:
            bool: True if a system prompt exists, False otherwise.
        """

        return any(message.get("role") == "system" for message in messages)

    @staticmethod
    def swap_system_prompt(
        system_prompt: dict = None, messages: Union[str, list[dict[str, str]]] = None
    ):
        """Swaps the system prompt with the system_prompt.

        Args:
            system_prompt: dict: The system prompt to swap.
            messages: Union[str, list[dict[str, str]]]: The messages to swap.

        Returns:
            list[dict[str, str]]: The messages with the system prompt swapped.
        """

        messages = Client.format_messages(messages)

        for message in messages:
            # Check if a system message exists
            if message.get("role") == "system":
                # If a system message exists, swap it with the system_prompt
                message = system_prompt
                # Move the system_prompt to the beginning of the list
                messages.insert(0, message)
                # Remove the system_prompt from its original position
                messages.remove(message)
                break

            else:
                # If no system message exists, add the system_prompt to the beginning of the list
                messages.insert(0, system_prompt)
                break

        # Remove any duplicate system messages
        while (
            len([message for message in messages if message.get("role") == "system"])
            > 1
        ):
            messages.pop()

        return messages

    @staticmethod
    def repair_messages(
        messages: list[dict], verbose: Optional[bool] = False
    ) -> list[dict]:
        """
        Repairs the messages by performing quick logic steps. Does not
        raise exceptions, attempts to fix the messages in a best effort manner.

        Args:
            messages: list[dict]: The messages to repair.
            verbose: Optional[bool]: Whether to log the repair process.

        Returns:
            list[dict]: The repaired messages.
        """

        # Ensure no item in the list is a nested list.
        if any(isinstance(message, list) for message in messages):
            messages = [item for sublist in messages for item in sublist]
            if verbose:
                print(f"Detected nested lists and flattened the list.")

        # Ensure messages are in the role user -> role assistant order &
        # repair order if items are mixmatched
        for i in range(len(messages) - 1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
                if (
                    not isinstance(messages[i + 1], dict)
                    or messages[i + 1].get("role") != "user"
                ):
                    messages[i + 1] = {"role": "user", "content": ""}
                    if verbose:
                        print(f"Detected a mixmatch in message order, repaired order.")
            elif isinstance(messages[i], dict) and messages[i].get("role") == "user":
                if (
                    not isinstance(messages[i + 1], dict)
                    or messages[i + 1].get("role") != "assistant"
                ):
                    messages[i + 1] = {"role": "assistant", "content": ""}
                    if verbose:
                        print(f"Detected a mixmatch in message order, repaired order.")

        return messages

    @staticmethod
    def add_messages(
        inputs: Union[str, list[dict], dict] = None,
        messages: list[dict] = None,
        type: Optional[Literal["user", "system", "assistant"]] = "user",
        verbose: Optional[bool] = False,
    ) -> list[dict]:
        """
        Adds a message to the thread, based on the type of message to add; and
        after performing some basic checks.

        Args:
            inputs: Union[str, list[dict], dict]: The messages to add.
            messages: list[dict]: The existing messages.
            type: Optional[Literal["user", "system", "assistant"]]: The type of message to add.
            verbose: Optional[bool]: Whether to log the addition of the message.

        Returns:
            list[dict]: The messages with the added message(s).
        """

        if isinstance(inputs, str):
            formatted_message = Client.format_messages(
                messages=inputs, verbose=verbose, type=type
            )

            messages.extend(formatted_message)

        elif isinstance(inputs, dict):
            messages.append(inputs)

        elif isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, dict):
                    messages.append(item)
                else:
                    if verbose:
                        print(f"Skipping invalid message format: {item}")

        return Client.repair_messages(messages, verbose)

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        provider: Optional[Literal["openai", "litellm"]] = None,
        verbose: bool = False,
    ):
        """Initializes the completion client with the specified parameters.

        Example:

            ```python
            client = Client(
                api_key = "sk-...",
                base_url = "https://api.openai.com/v1",
                organization = "org-...",
                provider = "openai",
                verbose = True
            )
            ```

        Args:
            api_key: Optional[str]: The API key to use for the client.
            base_url: Optional[str]: The base URL to use for the client.
            organization: Optional[str]: The organization to use for the client.
            provider: Optional[Literal["openai", "litellm"]]: The provider to use for the client.
            verbose: bool: Whether to log the client initialization process.

        Returns:
            None
        """

        self.clients = ClientProviders()

        self.config = ClientConfig(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            verbose=verbose,
        )

        self.provider = provider

        if self.provider:
            self.clients.client = self.__init_client__()

    def __init_client__(self):
        """
        Initializes the specified client library.
        """

        print(self.provider)

        if self.provider == "openai":
            from openai import OpenAI

            client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization,
            )

        elif self.provider == "litellm":
            import litellm
            from litellm import LiteLLM

            litellm.drop_params = True

            client = LiteLLM(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization,
            )

        if self.config.verbose:
            logger.info(f"Initialized {self.provider} client")

        return client

    def __patch_client__(self, mode: Optional[InstructorMode] = "tool_call"):
        """
        Patches the client with Instructor.
        """

        if not self.clients.client:
            logger.critical("Client not initialized. Please specify a valid provider.")
            raise

        if self.provider == "openai":
            from instructor import from_openai

            patched_client = from_openai(
                self.clients.client, mode=get_mode(mode) if mode else None
            )

        else:
            from instructor import patch

            patched_client = patch(
                self.clients.client, mode=get_mode(mode) if mode else None
            )

        if self.config.verbose:
            logger.info(f"Patched {self.provider} client with Instructor")

        return patched_client

    def chat_completion(self, args: CompletionArgs):
        """
        Runs a standard chat completion.

        Args:
            args: CompletionArgs: The arguments to the completion.

        Returns:
            CompletionResponse: The response to the completion.
        """

        exclude_params = {"response_model"}

        try:
            if args.tools is None:
                exclude_params.update({"tools", "parallel_tool_calls", "tool_choice"})

            # O1 Specific Handler
            # Will be removed once OpenAI supports all O1 Parameters
            if args.model.startswith("o1-"):
                logger.warning(
                    "OpenAI O1- model detected. Removing all non-supported parameters."
                )
                exclude_params.update(
                    {
                        "max_tokens",
                        "temperature",
                        "top_p",
                        "frequency_penalty",
                        "presence_penalty",
                        "stop",
                        "tools",
                        "tool_choice",
                        "parallel_tool_calls",
                    }
                )

            if args.stream:
                if self.config.verbose:
                    logger.info(f"Streaming completion... with {args.model} model")

                stream = self.clients.client.chat.completions.create(
                    **args.model_dump(exclude=exclude_params)
                )
                return (
                    chunk.choices[0].delta.content
                    for chunk in stream
                    if chunk.choices[0].delta.content
                )

            else:
                if self.config.verbose:
                    logger.info(f"Generating completion... with {args.model} model")

                exclude_params.add("stream")

                return self.clients.client.chat.completions.create(
                    **args.model_dump(exclude=exclude_params)
                )

        except Exception as e:
            logger.error(f"Error in chat_completion: {e}")
            raise

    def instructor_completion(self, args: CompletionArgs):
        """Runs an Instructor completion

        Args:
            args: CompletionArgs: The arguments to the completion.

        Returns:
            CompletionResponse: The response to the completion.
        """

        exclude_params = set()

        try:
            if not self.clients.instructor:
                self.clients.instructor = self.__patch_client__()

            if args.tools is None:
                exclude_params.update({"tools", "parallel_tool_calls", "tool_choice"})

            if self.config.verbose:
                logger.info(f"Excluding the following parameters: {exclude_params}")
                logger.info(f"Args: {args.model_dump(exclude=exclude_params)}")

            if args.model.startswith("o1-"):
                logger.warning(
                    "OpenAI O1- model detected. Removing all non-supported parameters."
                )
                exclude_params.update(
                    {
                        "max_tokens",
                        "temperature",
                        "top_p",
                        "frequency_penalty",
                        "presence_penalty",
                        "tools",
                        "parallel_tool_calls",
                        "tool_choice",
                        "stop",
                    }
                )

            if args.stream:
                if self.config.verbose:
                    logger.info(
                        f"Streaming Instructor completion... with {args.model} model"
                    )

                exclude_params.add("stream")

                return self.clients.instructor.chat.completions.create_partial(
                    **args.model_dump(exclude=exclude_params)
                )

            else:
                if self.config.verbose:
                    logger.info(
                        f"Generating Instructor completion... with {args.model} model"
                    )

                exclude_params.add("stream")

                return self.clients.instructor.chat.completions.create(
                    **args.model_dump(exclude=exclude_params)
                )

        except Exception as e:
            logger.error(f"Error in instructor_completion: {e}")
            raise

    def execute_tool_call(
        self,
        tools: List[Tool],
        args: CompletionArgs,
        response: CompletionResponse,
    ) -> Union[CompletionResponse, CompletionArgs, None]:
        """Executes the tool calls. if run tools is True.

        Args:
            tools: List[Tool]: The tools to execute.
            args: CompletionArgs: The arguments to the completion.
            response: CompletionResponse: The response to the completion.

        Returns:
            CompletionResponse: The response to the completion.
        """

        if not response.choices[0].message.tool_calls:
            return None

        if response.choices[0].message.tool_calls:
            args.messages.append(response.choices[0].message.model_dump())

            tools_executed = False
            for tool_call in response.choices[0].message.tool_calls:
                tool = next(
                    (t for t in tools if t.name == tool_call.function.name), None
                )
                if isinstance(tool.tool, Callable):
                    if self.config.verbose:
                        logger.info(
                            f"Executing tool {tool.name} with arguments {tool_call.function.arguments}"
                        )

                    try:
                        tool_response = tool.tool(
                            **json.loads(tool_call.function.arguments)
                        )
                        tool_call_result_message = {
                            "role": "tool",
                            "content": str(tool_response),
                            "tool_call_id": tool_call.id,
                        }
                        args.messages.append(tool_call_result_message)
                        tools_executed = True
                    except Exception as e:
                        logger.error(f"Error executing tool {tool.name}: {e}")
                        raise e
                else:
                    logger.warning(
                        f"Tool {tool_call.function.name} was called but not found or not callable."
                    )

            if tools_executed:
                return args

            else:
                logger.warning("Tools were called, but none were executed.")

        return response

    def run_completion(
        self,
        messages: Union[str, list[dict]] = None,
        model: str = "gpt-4o",
        response_model: Optional[Type[BaseModel]] = None,
        mode: Optional[InstructorMode] = "tool_call",
        max_retries: Optional[int] = 3,
        run_tools: Optional[bool] = True,
        tools: Optional[List[ToolType]] = None,
        parallel_tool_calls: Optional[bool] = False,
        tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = False,
    ):
        """
        Runs a completion with the specified arguments.

        Example:
            ```python
            completion(
                messages = "Hello!",
                model = "gpt-4o-mini
            )
            ```

        Args:
            messages: Union[str, list[dict]]: The messages to complete.
            model: str: The model to use.
            response_model: Optional[Type[BaseModel]]: The Pydantic model to use.
            mode: Optional[InstructorMode]: The Instructor mode to use.
            max_retries: Optional[int]: The maximum number of retries to use.
            run_tools: Optional[bool]: Whether to run tools.
            tools: Optional[List[ToolType]]: The tools to use.
            parallel_tool_calls: Optional[bool]: Whether to run tool calls in parallel.
            tool_choice: Optional[Literal["none", "auto", "required"]]: Whether to run tool calls in parallel.
            max_tokens: Optional[int]: The maximum number of tokens to use.
            temperature: Optional[float]: The temperature to use.
            top_p: Optional[float]: The top p to use.
            frequency_penalty: Optional[float]: The frequency penalty to use.
            presence_penalty: Optional[float]: The presence penalty to use.
            stop: Optional[List[str]]: The stop to use.
            stream: Optional[bool]: Whether to stream the completion.

        Returns:
            CompletionResponse: The completion response.
        """

        formatted_tools = None
        if tools:
            formatted_tools = self.format_to_openai_tools(tools)

        args = CompletionArgs(
            messages=self.format_messages(messages),
            model=model,
            response_model=response_model,
            tools=self.get_tool_dict(formatted_tools) if formatted_tools else None,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
        )

        if not response_model:
            if not run_tools or not formatted_tools:
                return self.chat_completion(args)

            else:
                args.stream = False
                base_response = self.chat_completion(args)

                args = self.execute_tool_call(formatted_tools, args, base_response)

                if args:
                    args.stream = stream

                    if self.config.verbose:
                        logger.info(f"Re-running completion with tools executed...")

                    return self.chat_completion(args)
                else:
                    return base_response

        else:
            if formatted_tools:
                original_args = args

                args.response_model = None
                args.stream = False
                base_response = self.chat_completion(args)

                args = self.execute_tool_call(formatted_tools, args, base_response)

                if args:
                    original_args.messages.extend(
                        args.messages[len(original_args.messages) :]
                    )
                    original_args.response_model = response_model
                    original_args.stream = stream
                    return self.instructor_completion(original_args)
                else:
                    return self.instructor_completion(original_args)

            else:
                return self.instructor_completion(args)

    def completion(
        self,
        messages: Union[str, list[dict]] = None,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
        mode: Optional[InstructorMode] = "tool_call",
        max_retries: Optional[int] = 3,
        run_tools: Optional[bool] = True,
        tools: Optional[List[ToolType]] = None,
        parallel_tool_calls: Optional[bool] = False,
        tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        provider: Optional[Literal["openai", "litellm"]] = None,
        verbose: bool = False,
    ):
        """
        Runs a completion with the specified arguments.

        Example:
            ```python
            completion(
                messages = "Hello!",
                model = "gpt-4o-mini
            )
            ```

        Args:
            messages: Union[str, list[dict]]: The messages to complete.
            model: str: The model to use.
            api_key : Optional[str]: The API key to use.
            base_url : Optional[str]: The base URL to use.
            organization : Optional[str]: The organization to use.
            response_model: Optional[Type[BaseModel]]: The Pydantic model to use.
            mode: Optional[InstructorMode]: The Instructor mode to use.
            max_retries: Optional[int]: The maximum number of retries to use.
            run_tools: Optional[bool]: Whether to run tools.
            tools: Optional[List[ToolType]]: The tools to use.
            parallel_tool_calls: Optional[bool]: Whether to run tool calls in parallel.
            tool_choice: Optional[Literal["none", "auto", "required"]]: Whether to run tool calls in parallel.
            max_tokens: Optional[int]: The maximum number of tokens to use.
            temperature: Optional[float]: The temperature to use.
            top_p: Optional[float]: The top p to use.
            frequency_penalty: Optional[float]: The frequency penalty to use.
            presence_penalty: Optional[float]: The presence penalty to use.
            stop: Optional[List[str]]: The stop to use.
            stream: Optional[bool]: Whether to stream the completion.
            provider : Optional[Literal["openai", "litellm"]]: The provider to use.
            verbose : bool: Whether to print verbose output.

        Returns:
            CompletionResponse: The completion response.
        """
        (
            recommended_provider,
            recommended_model,
            recommended_base_url,
            recommended_api_key,
        ) = self.recommend_client_by_model(model, base_url, api_key)

        if self.config.verbose:
            logger.info(f"Recommended Provider: {recommended_provider}")
            logger.info(f"Recommended Model: {recommended_model}")
            logger.info(f"Recommended Base URL: {recommended_base_url}")

        # Reinitialize client only if the recommended provider is different
        if recommended_provider != self.provider:
            self.__init__(
                api_key=recommended_api_key or api_key or self.config.api_key,
                base_url=recommended_base_url or base_url or self.config.base_url,
                organization=organization or self.config.organization,
                provider=recommended_provider,
                verbose=verbose or self.config.verbose,
            )

        # Update model if it was changed by recommend_client_by_model
        if model != recommended_model:
            model = recommended_model

        if response_model:
            mode = get_mode(mode)

            if model.startswith("o1-"):
                logger.warning(
                    "OpenAI O1- model detected. Using JSON_O1 Instructor Mode."
                )
                mode = Mode.JSON_O1

            if not self.clients.instructor:
                self.clients.instructor = self.__patch_client__(mode)

            else:
                self.clients.instructor.mode = mode

            if verbose:
                logger.info(f"Instructor Mode: {self.clients.instructor.mode}")

        return self.run_completion(
            messages=messages,
            model=model,
            response_model=response_model,
            mode=mode,
            max_retries=max_retries,
            run_tools=run_tools,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
        )

    @staticmethod
    def _completion(
        messages: Union[str, list[dict]] = None,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
        mode: Optional[InstructorMode] = "tool_call",
        max_retries: Optional[int] = 3,
        run_tools: Optional[bool] = True,
        tools: Optional[List[ToolType]] = None,
        parallel_tool_calls: Optional[bool] = False,
        tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        provider: Optional[Literal["openai", "litellm"]] = None,
        verbose: bool = False,
    ):
        """
        Runs a completion with the specified arguments.

        Example:
            ```python
            completion(
                messages = "Hello!",
                model = "gpt-4o-mini
            )
            ```

        Args:
            messages: Union[str, list[dict]]: The messages to complete.
            model: str: The model to use.
            api_key : Optional[str]: The API key to use.
            base_url : Optional[str]: The base URL to use.
            organization : Optional[str]: The organization to use.
            response_model: Optional[Type[BaseModel]]: The Pydantic model to use.
            mode: Optional[InstructorMode]: The Instructor mode to use.
            max_retries: Optional[int]: The maximum number of retries to use.
            run_tools: Optional[bool]: Whether to run tools.
            tools: Optional[List[ToolType]]: The tools to use.
            parallel_tool_calls: Optional[bool]: Whether to run tool calls in parallel.
            tool_choice: Optional[Literal["none", "auto", "required"]]: Whether to run tool calls in parallel.
            max_tokens: Optional[int]: The maximum number of tokens to use.
            temperature: Optional[float]: The temperature to use.
            top_p: Optional[float]: The top p to use.
            frequency_penalty: Optional[float]: The frequency penalty to use.
            presence_penalty: Optional[float]: The presence penalty to use.
            stop: Optional[List[str]]: The stop to use.
            stream: Optional[bool]: Whether to stream the completion.
            provider : Optional[Literal["openai", "litellm"]]: The provider to use.
            verbose : bool: Whether to print verbose output.

        Returns:
            CompletionResponse: The completion response.
        """

        if provider:
            client = Client(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                provider=provider,
                verbose=verbose,
            )

        else:
            provider, model, base_url, api_key = Client.recommend_client_by_model(model)

            client = Client(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                provider=provider,
                verbose=verbose,
            )

        if response_model:
            if not client.clients.instructor:
                client.clients.instructor = client.__patch_client__()

            mode = get_mode(mode)

            if model.startswith("o1-"):
                logger.warning(
                    "OpenAI O1- model detected. Using JSON_O1 Instructor Mode."
                )
                mode = Mode.JSON_O1

            client.clients.instructor.mode = mode

            if verbose:
                logger.info(f"Instructor Mode: {mode}")

        return client.run_completion(
            messages=messages,
            model=model,
            response_model=response_model,
            mode=mode,
            max_retries=max_retries,
            run_tools=run_tools,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
        )


def completion(
    messages: Union[str, list[dict]] = None,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    response_model: Optional[Type[BaseModel]] = None,
    mode: Optional[InstructorMode] = "tool_call",
    max_retries: Optional[int] = 3,
    run_tools: Optional[bool] = True,
    tools: Optional[List[ToolType]] = None,
    parallel_tool_calls: Optional[bool] = False,
    tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    client: Optional[Literal["openai", "litellm"]] = None,
    verbose: bool = False,
) -> CompletionResponse:
    """Runs an LLM completion, with tools, streaming or Pydantic structured outputs.

    Example:

        ```python
        completion(
            messages = messages,
            model = model,
            api_key = api_key,
            base_url = base_url,
            organization = organization,
        )
        ```

    Args:
        messages: Union[str, list[dict]]: The messages to complete.
        model: str: The model to use.
        api_key : Optional[str]: The API key to use.
        base_url : Optional[str]: The base URL to use.
        organization : Optional[str]: The organization to use.
        response_model: Optional[Type[BaseModel]]: The Pydantic model to use.
        mode: Optional[InstructorMode]: The Instructor mode to use.
        max_retries: Optional[int]: The maximum number of retries to use.
        run_tools: Optional[bool]: Whether to run tools.
        tools: Optional[List[ToolType]]: The tools to use.
        parallel_tool_calls: Optional[bool]: Whether to run tool calls in parallel.
        tool_choice: Optional[Literal["none", "auto", "required"]]: Whether to run tool calls in parallel.
        max_tokens: Optional[int]: The maximum number of tokens to use.
        temperature: Optional[float]: The temperature to use.
        top_p: Optional[float]: The top p to use.
        frequency_penalty: Optional[float]: The frequency penalty to use.
        presence_penalty: Optional[float]: The presence penalty to use.
        stop: Optional[List[str]]: The stop to use.
        stream: Optional[bool]: Whether to stream the completion.
        provider : Optional[Literal["openai", "litellm"]]: The provider to use.
        verbose : bool: Whether to print verbose output.

    Returns:
        CompletionResponse: The completion response.
    """

    provider = client

    return Client._completion(
        messages=messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        response_model=response_model,
        mode=mode,
        max_retries=max_retries,
        run_tools=run_tools,
        tools=tools,
        parallel_tool_calls=parallel_tool_calls,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        stream=stream,
        provider=provider,
        verbose=verbose,
    )


if __name__ == "__main__":

    class PersonModel(BaseModel):
        secret_identity: str
        name: str
        age: int

    def get_secret_identity(name: str):
        return "Batman"

    print(completion("Who is SpiderMan", verbose=True, response_model=PersonModel))

    print(
        completion(
            messages="Who is SpiderMan",
            verbose=True,
            base_url="http://localhost:11434",
            model = "ollama/bespoke-minicheck"
        )
    )
