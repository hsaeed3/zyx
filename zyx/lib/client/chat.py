from ... import logger
from ...lib.types import client as clienttypes
import json
from pydantic import BaseModel
from typing import (
    Any,
    Callable,
    Generator,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union
)


class ChatClient:

    """Base class for all LLM completions, and functions. Runs
    either the LiteLLM or OpenAI clients.
    
    Args:
        client: Literal["openai", "litellm"]: The client to use.
        api_key: Optional[str]: The API key to use.
        base_url: Optional[str]: The base URL to use.
        organization: Optional[str]: The organization to use.
        verbose: bool: Whether to log the client initialization process.
    """

    def __init__(
        self,
        client: Literal["openai", "litellm"] = "openai",
        api_key : Optional[str] = None,
        base_url : Optional[str] = None,
        organization : Optional[str] = None,
        verbose : bool = False
    ):
        
        self.config = clienttypes.ClientConfig(
            client = client,
            api_key = api_key,
            base_url = base_url,
            organization = organization,
            verbose = verbose
        )

        self.providers = self.__init_client__()

        pass


    def __init_client__(self) -> clienttypes.ClientProviders:

        """Returns the initalized client and instructor patch."""

        try:
            if self.config.client == "openai":
                from openai import OpenAI
                from instructor import from_openai

                client = OpenAI(
                    api_key = self.config.api_key,
                    base_url = self.config.base_url,
                    organization = self.config.organization
                )

                patch = from_openai(client)
            
            else:
                import litellm
                from litellm import LiteLLM
                from instructor import patch as instructor_patch

                litellm.drop_params = True

                client = LiteLLM(
                    api_key = self.config.api_key,
                    base_url = self.config.base_url,
                    organization = self.config.organization
                )
                
                patch = instructor_patch(client)

        except Exception as e:
            logger.error(f"Error initializing client: {e}")

        if self.config.verbose:
            logger.info(f"Initialized {self.config.client} client.")

        return clienttypes.ClientProviders(
            client = client,
            instructor = patch
        )
    

    def __self_destruct__(self):
        """Deletes the client and every single thing associated with it.
        Provided by Dr. Hienz Doofenshmirtz."""

        self = None


    @staticmethod
    def _get_tool_dict(
        tools : List[clienttypes.Tool]
    ) -> List[Dict[str, Any]]:
        
        """Converts the tools to a list of dictionaries.
        
        Args:
            tools: List[clienttypes.Tool]: The tools to convert.

        Returns:
            List[Dict[str, Any]]: The converted tools.
        """

        tool_dict = []
        for tool in tools:
            tool_dict.append(tool.openai_tool)

        return tool_dict


    @staticmethod
    def _recommend_client_by_model(
        model : str, base_url : Optional[str] = None, api_key : Optional[str] = None
    ) -> tuple[Literal["openai", "litellm"], Optional[str], Optional[str], Optional[str]]:
        
        """Recommends the client to use for the given model. Used in one-shot completions.
        
        Args:
            model: str: The model to recommend the client for.
            base_url: Optional[str]: The base URL to use for the client.
            api_key: Optional[str]: The API key to use for the client.

        Returns:
            tuple[Literal["openai", "litellm"], Optional[str], Optional[str], Optional[str]]: The recommended client, model, base URL, and API key.
        """
        
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
    def _format_messages(
        messages: Union[str, list[dict]] = None, 
        verbose : Optional[bool] = False,
        type : Optional[Literal["user", "system", "assistant"]] = "user"
        ) -> list[dict]:

        """Formats the messages into a list of dictionaries.
        
        Args:
            messages: Union[str, list[dict]]: The messages to format.
            verbose: bool: Whether to log the formatting process.

        Returns:
            list[dict]: The formatted messages.
        """

        if isinstance(messages, str):

            if verbose:
                logger.info(f"Converting string to message format.")

            return [{"role": str(type), "content": str(messages)}]
        
        elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):


            if verbose:
                logger.info(f"Messages are in the correct format.")

            return messages
        else:
            raise ValueError("Invalid message format")


    @staticmethod
    def _does_system_prompt_exist(messages: list[dict]) -> bool:

        """Simple boolean check to see if a system prompt exists in the messages.
        
        Args:
            messages: list[dict]: The messages to check.

        Returns:
            bool: True if a system prompt exists, False otherwise.
        """

        return any(message.get("role") == "system" for message in messages)

    
    @staticmethod
    def _swap_system_prompt(
        system_prompt: dict = None, messages: Union[str, list[dict[str, str]]] = None
    ):
        
        """Swaps the system prompt with the system_prompt.
        
        Args:
            system_prompt: dict: The system prompt to swap.
            messages: Union[str, list[dict[str, str]]]: The messages to swap.

        Returns:
            list[dict[str, str]]: The messages with the system prompt swapped.
        """

        messages = ChatClient.format_messages(messages)

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
        while len([message for message in messages if message.get("role") == "system"]) > 1:
            messages.pop()

        return messages


    def _format_to_openai_tools(self, tools : List[clienttypes.ToolType]) -> List[clienttypes.Tool]:

        """Converts the tools to a list of dictionaries.
        
        Args:
            tools: List[clienttypes.ToolType]: The tools to convert.

        Returns:
            List[clienttypes.Tool]: The converted tools.
        """

        from ...lib.utils.convert_to_openai_tool import convert_to_openai_tool

        formatted_tools = []
        for tool in tools:
            formatted_tool = convert_to_openai_tool(tool)
            formatted_tools.append(
                clienttypes.Tool(
                    name = formatted_tool["function"]["name"],
                    tool = tool,
                    openai_tool = formatted_tool
                )
            )

        if self.config.verbose:
            logger.info(f"Formatted {len(formatted_tools)} tools.")

        return formatted_tools
                

    def _execute_tool_call(
        self,
        tools: List[clienttypes.Tool],
        args: clienttypes.CompletionArgs,
        response: clienttypes.CompletionResponse
    ) -> Union[clienttypes.CompletionResponse, clienttypes.CompletionArgs, None]:

        """Executes the tool calls. if run tools is True.
        
        Args:
            tools: List[clienttypes.Tool]: The tools to execute.
            args: clienttypes.CompletionArgs: The arguments to the completion.
            response: clienttypes.CompletionResponse: The response to the completion.

        Returns:
            clienttypes.CompletionResponse: The response to the completion.
        """

        if not response.choices[0].message.tool_calls:
            return None

        if response.choices[0].message.tool_calls:
            args.messages.append(response.choices[0].message.model_dump())
            
            tools_executed = False
            for tool_call in response.choices[0].message.tool_calls:
                tool = next((t for t in tools if t.name == tool_call.function.name), None)
                if isinstance(tool.tool, Callable):
                    if self.config.verbose:
                        logger.info(f"Executing tool {tool.name} with arguments {tool_call.function.arguments}")

                    try:
                        tool_response = tool.tool(**json.loads(tool_call.function.arguments))
                        tool_call_result_message = {
                            "role": "tool",
                            "content": str(tool_response),
                            "tool_call_id": tool_call.id
                        }
                        args.messages.append(tool_call_result_message)
                        tools_executed = True
                    except Exception as e:
                        logger.error(f"Error executing tool {tool.name}: {e}")
                        raise e
                else:
                    logger.warning(f"Tool {tool_call.function.name} was called but not found or not callable.")

            if tools_executed:
                return args
            
            else:
                logger.warning("Tools were called, but none were executed.")

        return response


    def _chat_completion(
        self,
        args: clienttypes.CompletionArgs
    ) -> Union[Dict[str, str], Generator]:
        
        """
        Args:
            args: clienttypes.CompletionArgs: The arguments to the completion.

        Returns:
            Union[Dict[str, str], Generator]: The response to the completion.
        """
            
        try:
            exclude_params = {"response_model"}
            if args.tools is None:
                exclude_params.update({"tools", "tool_choice", "parallel_tool_calls"})
            if args.model.startswith("o1-"):
                logger.warning("Removing all unsupported parameters for OpenAI o1 or o1-mini")
                exclude_params.update({"max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "stop",
                                       "tools", "tool_choice", "parallel_tool_calls"})

            if args.stream:
                if self.config.verbose:
                    logger.info("Streaming response... \n")

                response = self.providers.client.chat.completions.create(
                    **args.model_dump(exclude=exclude_params)
                )
                return (chunk.choices[0].delta.content for chunk in response if chunk.choices[0].delta.content)
            else:
                if self.config.verbose:
                    logger.info("Generating response... \n")
                exclude_params.add("stream")
                response = self.providers.client.chat.completions.create(
                    **args.model_dump(exclude=exclude_params),
                    stream=False
                )
                return response

        except TypeError as e:
            logger.error(f"Serialization error in base chat completion: {e}. \n Args: {args}")
            raise e
        except Exception as e:
            logger.error(f"Error in base chat completion: {e}")
            raise e


    def _instructor_chat_completion(
        self,
        args: clienttypes.CompletionArgs
    ) -> Any:
        
        """Instructor chat completion.

        Args:
            args: clienttypes.CompletionArgs: The arguments to the completion.

        Returns:
            Any: The response to the completion.
        """

        try:
            exclude_params = {"stream", "response_model"}
            if args.tools is None:
                exclude_params.update({"tools", "tool_choice", "parallel_tool_calls"})
            if args.model.startswith("o1-"):
                logger.warning("Removing all unsupported parameters for OpenAI o1 or o1-mini")
                exclude_params.update({"max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "stop",
                                       "tools", "tool_choice", "parallel_tool_calls"})

            if args.stream:
                if self.config.verbose:
                    logger.info("Streaming instructor response... \n")
                response = self.providers.instructor.chat.completions.create_partial(
                    **args.model_dump(exclude=exclude_params),
                    response_model=args.response_model
                )
            else:
                if self.config.verbose:
                    logger.info("Generating instructor response... \n")
                response = self.providers.instructor.chat.completions.create(
                    **args.model_dump(exclude=exclude_params),
                    response_model=args.response_model
                )
            return response
        except Exception as e:
            logger.error(f"Error in instructor chat completion: {e}")
            raise e
        

    def completion(
            self,
            messages: Union[str, list[dict]] = None,
            model: str = "gpt-4o",
            response_model: Optional[Type[BaseModel]] = None,
            mode: Optional[clienttypes.InstructorMode] = "tool_call",
            max_retries : Optional[int] = 3,
            run_tools: Optional[bool] = True,
            tools: Optional[List[clienttypes.ToolType]] = None,
            parallel_tool_calls: Optional[bool] = False,
            tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            stop: Optional[List[str]] = None,
            stream: Optional[bool] = False
    ) -> Any:
        
        """Run an LLM completion, through any LiteLLM compatible model, using both either the OpenAI or LiteLLM API
        clients. Optional arguments for structured outputs with instructor, using `response_model` and `mode`. As
        well as tool calling & tool execution support.

        Example:

        ```python
        def add(a: int, b: int) -> int:
            "A function that adds two numbers"
            return a + b

        print(
            completion(
                "what is 2 + 2?",
                tools = [add]
            )
        )
        ```

        Args:
            messages: Union[str, list[dict]]: The messages to complete.
            model: str: The model to use.
            response_model: Optional[Type[BaseModel]]: The response model to use.
            mode: Optional[clienttypes.InstructorMode]: The mode to use.
            run_tools: Optional[bool]: Whether to run tools.
            tools: Optional[List[clienttypes.ToolType]]: The tools to use.
            parallel_tool_calls: Optional[bool]: Whether to run tool calls in parallel.
            tool_choice: Optional[Literal["none", "auto", "required"]]: Whether to run tool calls in parallel.
            max_tokens: Optional[int]: The maximum number of tokens to generate.
            temperature: Optional[float]: The temperature to use.
            top_p: Optional[float]: The top p to use.
            frequency_penalty: Optional[float]: The frequency penalty to use.
            presence_penalty: Optional[float]: The presence penalty to use.
            stop: Optional[List[str]]: The stop to use.
            stream: Optional[bool]: Whether to stream the response.
            
        Returns:
            Any: The response to the completion.
        """

        if isinstance(messages, str):
            messages = self._format_messages(messages, verbose=self.config.verbose)
        elif not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            logger.error(f"Invalid message format: {messages}")
            raise ValueError("Invalid message format")

        formatted_tools = None

        if tools:
            formatted_tools = self._format_to_openai_tools(tools)

        if self.config.verbose:
            logger.info(f"Running completion with {len(messages)} messages, with {model}.")

        args = clienttypes.CompletionArgs(
            messages=messages,
            model=model,
            response_model=response_model,
            mode=mode,
            tools=self._get_tool_dict(formatted_tools) if formatted_tools else None,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream
        )

        if not response_model:
            if not run_tools or not formatted_tools:
                return self._chat_completion(args)
            else:
                args.stream = False
                base_response = self._chat_completion(args)

                args = self._execute_tool_call(formatted_tools, args, base_response)

                if args:
                    args.stream = stream
                    return self._chat_completion(args)
                else:
                    return base_response
        else:
            if formatted_tools:
                original_args = args

                args.response_model = None
                args.stream = False
                base_response = self._chat_completion(args)
                args = self._execute_tool_call(formatted_tools, args, base_response)
                if args:
                    args.response_model = response_model
                    args.stream = stream
                    return self._instructor_chat_completion(args)
                else:
                    return self._instructor_chat_completion(original_args)
            else:
                return self._instructor_chat_completion(args)
            

def completion(
    messages: Union[str, list[dict]],
    model: str = "gpt-4o-mini",
    client : Literal["openai", "litellm"] = "openai",
    response_model: Optional[Type[BaseModel]] = None,
    mode: Optional[clienttypes.InstructorMode] = "tool_call",
    max_retries : Optional[int] = 3,
    api_key : Optional[str] = None,
    base_url : Optional[str] = None,
    organization : Optional[str] = None,
    run_tools: Optional[bool] = True,
    tools: Optional[List[clienttypes.ToolType]] = None,
    parallel_tool_calls: Optional[bool] = False,
    tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    verbose : Optional[bool] = False

) -> clienttypes.CompletionResponse:
    
    """
    Run an LLM chat completion, through any LiteLLM compatible model, using both either the OpenAI or LiteLLM API
    clients. Optional arguments for structured outputs with instructor, using `response_model` and `mode`. As
    well as tool calling & tool execution support.

    Example:

    ```python
    def add(a: int, b: int) -> int:
        "A function that adds two numbers"
        return a + b

    print(
        completion(
            "what is 2 + 2?",
            tools = [add]
        )
    )
    ```

    Args:
        messages: Union[str, list[dict]]: The messages to complete.
        model: str: The model to use.
        response_model: Optional[Type[BaseModel]]: The response model to use.
        mode: Optional[clienttypes.InstructorMode]: The mode to use.
        run_tools: Optional[bool]: Whether to run tools.
        tools: Optional[List[clienttypes.ToolType]]: The tools to use.
        parallel_tool_calls: Optional[bool]: Whether to run tool calls in parallel.
        tool_choice: Optional[Literal["none", "auto", "required"]]: Whether to run tool calls in parallel.
        max_tokens: Optional[int]: The maximum number of tokens to generate.
        temperature: Optional[float]: The temperature to use.
        top_p: Optional[float]: The top p to use.
        frequency_penalty: Optional[float]: The frequency penalty to use.
        presence_penalty: Optional[float]: The presence penalty to use.
        stop: Optional[List[str]]: The stop to use.
        stream: Optional[bool]: Whether to stream the response.
        
    Returns:
        Any: The response to the completion.
    """

    client_mode = client
    
    if client_mode == "openai":
        client_mode, model, base_url, api_key = ChatClient._recommend_client_by_model(
            model, base_url
        )
        if client_mode == "litellm":
            logger.info(
                "Defaulting to LiteLLM client."
            )

    client = ChatClient(
        client = client_mode,
        api_key = api_key,
        base_url = base_url,
        organization = organization,
        verbose = verbose
    )

    if model.startswith("o1-"):
        logger.info("Using JSON_O1 mode for o1-preview or o1-mini models.")
        mode = clienttypes.Mode.JSON_O1

    mode = clienttypes.get_mode(mode)

    client.providers.instructor.mode = mode

    return client.completion(
        messages = messages,
        model = model,
        response_model = response_model,
        mode = mode,
        run_tools = run_tools,
        tools = tools,
        parallel_tool_calls = parallel_tool_calls,
        tool_choice = tool_choice,
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
        frequency_penalty = frequency_penalty,
        presence_penalty = presence_penalty,
        stop = stop,
        stream = stream
    )


if __name__ == "__main__":

    print(completion(messages = "hi", model = "gpt-4o-mini", verbose = True))

    class Response(BaseModel):
        code : str
    

    def get_secret_code() -> str:
        """Returns the secret code"""
        return "banana"

    print(
        completion(
            "what is the secret code?",
            tools = [get_secret_code],
            model = "o1-mini",
            verbose = True
        )
    )