from pydantic import BaseModel
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)


# --- Models / Types ---


# Placeholder for ModelResponse Return Type
ModelResponse = Type["ModelResponse"]


# Optimizer Types
Optimizer = Literal["costar", "tidd-ec"]


# Base Response Type
Response = Union[ModelResponse, BaseModel, Generator]


# Instructor Parsing Mode Mapping
ClientModeParams = Literal["json", "json_schema", "md_json", "parallel", "tools"]


# Tool Parameter Model (For Internal Consistency)
class ClientToolParams(BaseModel):
    tools: List[Union[Callable, dict, BaseModel]] = None
    openai_tools: List[dict] = None
    mapping: Optional[dict] = None


# Client Parameter Model (For Internal Consistency)
class ClientParams(BaseModel):
    messages: Union[str, list[dict]] = None

    model: Optional[str] = "gpt-4o-mini"
    tools: Optional[ClientToolParams] = None
    run_tools: Optional[bool] = True
    response_model: Optional[Union[Any, BaseModel]] = None
    mode: Optional[ClientModeParams] = "tools"

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    organization: Optional[str] = None

    top_p: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 3
    max_retries: Optional[int] = None
    kwargs: Optional[dict] = None


# --- Client ---


class CompletionClient:
    def __init__(
        self,
    ):
        pass

    @staticmethod
    def format_messages(messages: Union[str, list[dict]] = None) -> list[dict]:
        """Simple str -> list[dict] formatter for messages."""
        try:
            # If the messages are a string, convert them to a list of messages
            if isinstance(messages, str):
                return [{"role": "user", "content": messages}]

            # If the messages are a list of dictionaries, return them
            elif isinstance(messages, list) and all(
                isinstance(m, dict) for m in messages
            ):
                return messages
            else:
                raise ValueError("Invalid message format")
        except Exception as e:
            print(f"Error formatting messages: {e}")
            return []

    @staticmethod
    def format_tools(tools: List[Union[Callable, dict, BaseModel]] = None):
        """Runs tool conversion to OpenAI mapping."""
        from .utils.tool_calling import convert_to_openai_tool

        try:
            if not tools:
                return None

            # Create Parameter Object
            params = ClientToolParams(tools=tools, openai_tools=[], mapping={})

            for tool in tools:
                # If the tool is a dictionary, add it to the openai_tools list
                if isinstance(tool, dict):
                    params.openai_tools.append(tool)

                # If the tool is a BaseModel or Callable, convert it to an OpenAI tool and add it to the openai_tools list
                elif isinstance(tool, BaseModel) or isinstance(tool, Callable):
                    params.openai_tools.append(convert_to_openai_tool(tool))

                # If the tool is not a dictionary, BaseModel, or Callable, raise an error
                else:
                    raise ValueError(f"Invalid tool type: {type(tool)}")

            # Create a mapping of the tool name to the tool itself
            for tool in tools:
                if isinstance(tool, Callable) or isinstance(tool, BaseModel):
                    params.mapping[tool.__name__] = tool
            return params
        except Exception as e:
            print(f"Error formatting tools: {e}")
            return None

    def run_base_completion(self):
        try:
            from litellm.main import completion

            # Run a standard LiteLLM completion
            return completion(
                model=self.params.model,
                messages=self.params.messages,
                base_url=self.params.base_url,
                tools=self.params.tools.openai_tools if self.params.tools else None,
                api_key=self.params.api_key,
                organization=self.params.organization,
                top_p=self.params.top_p,
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
                **self.params.kwargs,
            )
        except Exception as e:
            print(f"Error running base completion: {e}")
            return None

    def stream_completion(self):
        """Streams Basic Text Response"""
        try:
            from litellm.main import completion

            # Run a streamed LiteLLM completion
            response = completion(
                model=self.params.model,
                messages=self.params.messages,
                base_url=self.params.base_url,
                tools=self.params.tools.openai_tools if self.params.tools else None,
                api_key=self.params.api_key,
                organization=self.params.organization,
                top_p=self.params.top_p,
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
                stream=True,
                **self.params.kwargs,
            )

            for chunk in response:
                yield chunk.choices[0].delta.content or ""

        except Exception as e:
            print(f"Error in stream_completion: {e}")
            yield None

    def run_instructor_completion(self):
        """Runs Structured Pydnatic Completion"""
        try:
            from litellm.main import completion
            from instructor import from_litellm, Mode

            if self.params.mode:
                if self.params.mode == "json":
                    mode = Mode.JSON
                elif self.params.mode == "json_schema":
                    mode = Mode.JSON_SCHEMA
                elif self.params.mode == "md_json":
                    mode = Mode.MD_JSON
                elif self.params.mode == "parallel":
                    mode = Mode.PARALLEL_TOOLS
                elif self.params.mode == "tools":
                    mode = Mode.TOOLS

            client = from_litellm(completion, mode=mode if mode else None)
            return client.chat.completions.create(
                messages=self.params.messages,
                model=self.params.model,
                response_model=self.params.response_model,
                base_url=self.params.base_url,
                api_key=self.params.api_key,
                tools=self.params.tools.openai_tools if self.params.tools else None,
                organization=self.params.organization,
                top_p=self.params.top_p,
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
                max_retries=self.params.max_retries,
                **self.params.kwargs,
            )
        except Exception as e:
            print(f"Error running instructor completion: {e}")
            return None

    def execute_tools(self, tool_calls: list[dict]):
        """Executes Tool Calls (Dependent on run_tools)"""
        try:
            import json

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                if function_name not in self.params.tools.mapping:
                    self.params.messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": "Tool not executable",
                        }
                    )
                    continue
                function_args = json.loads(tool_call.function.arguments)
                tool_result = self.params.tools.mapping[function_name](**function_args)
                self.params.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result),
                    }
                )

            return self.params.messages
        except Exception as e:
            print(f"Error executing tools: {e}")
            return self.params.messages

    def optimize_system_prompt(self, optimize: Optimizer, verbose: bool = False):
        from .utils.messages import swap_system_prompt, does_system_prompt_exist
        from .llm.create_system_prompt import create_system_prompt
        from .llm.optimize_system_prompt import optimize_system_prompt

        if verbose:
            print("Optimizing System Prompt...")

        try:
            if not does_system_prompt_exist(self.params.messages):
                user_message = str(self.params.messages[-1]["content"])
                prompt = create_system_prompt(
                    instructions=user_message,
                    type=optimize,
                    model=self.params.model,
                    api_key=self.params.api_key,
                    base_url=self.params.base_url,
                    temperature=self.params.temperature,
                    response_format="dict",
                    verbose=verbose,
                )
            else:
                prompt = optimize_system_prompt(
                    prompt=self.params.messages,
                    type=optimize,
                    model=self.params.model,
                    api_key=self.params.api_key,
                    base_url=self.params.base_url,
                    temperature=self.params.temperature,
                    response_format="dict",
                    verbose=verbose,
                )

        except Exception as e:
            print(f"Error optimizing system prompt: {e}")
            return self.params.messages

        return swap_system_prompt(system_prompt=prompt, messages=self.params.messages)

    def interpret_images(self, image_urls: Union[str, List[str]], model: str) -> str:
        """Interprets images using the specified vision model."""
        if isinstance(image_urls, str):
            image_urls = [image_urls]

        image_content = [{"type": "text", "text": "What's in this image?"}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ]

        response = completion(
            model=model,
            messages=[{"role": "user", "content": image_content}],
        )

        return response.choices[0].message.content

    def completion(
        self,
        messages: Union[str, list[dict]] = None,
        model: Optional[str] = "gpt-4o-mini",
        response_model: Optional[BaseModel] = None,
        mode: Optional[ClientModeParams] = "md_json",
        optimize: Union[Optimizer, None] = None,
        tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
        run_tools: Optional[bool] = True,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = 3,
        params: Optional[ClientParams] = None,
        verbose: Optional[bool] = False,
        stream: Optional[bool] = False,
        image_urls: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Response:
        """Runs a tool calling & structured output capable completion or completion chain.

        Example:
            ```python

            import zyx

            zyx.completion()
            ```

        Parameters:
            - messages (Union[str, list[dict]]): The messages to send to the model.
            - model (Optional[str]): The model to use for completion.
            - response_model (Optional[BaseModel]): The response model to use for completion.
            - mode (Optional[ClientModeParams]): The mode to use for completion.
            - optimize (Optional[Literal["costar"]]): The optimization to use for completion.
            - tools (Optional[List[Union[Callable, dict, BaseModel]]]): The tools to use for completion.
            - run_tools (Optional[bool]): Whether to run the tools.
            - base_url (Optional[str]): The base URL to use for completion.
            - api_key (Optional[str]): The API key to use for completion.
            - organization (Optional[str]): The organization to use for completion.
            - top_p (Optional[float]): The top p value to use for completion.
            - temperature (Optional[float]): The temperature value to use for completion.
            - max_tokens (Optional[int]): The maximum tokens to use for completion.
            - max_retries (Optional[int]): The maximum retries to use for completion.
            - params (Optional[ClientParams]): The parameters to use for completion.
            - verbose (Optional[bool]): Whether to print the parameters.
            - stream (Optional[bool]): Whether to stream the response.
            - image_urls (Optional[Union[str, List[str]]]): The image URLs to interpret.

        Returns:
            - The completion response.
        """
        try:
            if params:
                self.params = params
            else:
                self.params = ClientParams(
                    messages=self.format_messages(messages),
                    model=model,
                    tools=self.format_tools(tools),
                    run_tools=run_tools,
                    response_model=response_model,
                    mode=mode,
                    base_url=base_url,
                    api_key=api_key,
                    organization=organization,
                    top_p=top_p,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    kwargs=kwargs,
                )

            if verbose:
                print(f"{self.params}\n")

            if optimize:
                self.params.messages = self.optimize_system_prompt(
                    optimize=optimize, verbose=verbose
                )

            if image_urls:
                if response_model or (self.params.tools and self.params.run_tools):
                    # Only interpret images if response_model or tools are provided
                    image_interpretation = self.interpret_images(image_urls, model)
                    self.params.messages.append(
                        {"role": "assistant", "content": image_interpretation}
                    )
                else:
                    # If only images are provided, format the message with image content
                    if isinstance(image_urls, str):
                        image_urls = [image_urls]
                    image_content = [
                        {
                            "type": "text",
                            "text": self.params.messages[-1]["content"]
                            if self.params.messages
                            else "What's in this image?",
                        }
                    ] + [
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in image_urls
                    ]
                    self.params.messages = [{"role": "user", "content": image_content}]

            if stream and not response_model:
                return self.stream_completion()
            else:
                if not self.params.tools or not self.params.run_tools:
                    if not self.params.response_model:
                        if verbose:
                            print("Running base completion")
                        return self.run_base_completion()
                    else:
                        if verbose:
                            print("Running instructor completion")
                        return self.run_instructor_completion()

                response = self.run_base_completion()

                if not self.params.response_model:
                    if not response.choices[0].message.tool_calls:
                        print("No tool calls. Returning Response...")
                        return response
                    else:
                        while response.choices[0].message.tool_calls:
                            message = response.choices[0].message
                            self.params.messages.append(message)
                            self.params.messages = self.execute_tools(
                                message.tool_calls
                            )
                            response = self.run_base_completion()
                        return response
                else:
                    if not response.choices[0].message.tool_calls:
                        self.params.messages.append(response.choices[0].message)
                        self.params.messages.append(
                            {
                                "role": "user",
                                "content": "Append tool responses to BaseModel.",
                            }
                        )
                        self.params.tools = None
                        return self.run_instructor_completion()
                    else:
                        while response.choices[0].message.tool_calls:
                            self.params.messages.append(response.choices[0].message)
                            self.params.messages = self.execute_tools(
                                response.choices[0].message.tool_calls
                            )
                            response = self.run_base_completion()
                        self.params.messages.append(
                            {
                                "role": "user",
                                "content": "Append tool responses to BaseModel.",
                            }
                        )
                        self.params.tools = None
                        return self.run_instructor_completion()
        except Exception as e:
            print(f"Error in completion: {e}")
            return None


def completion(
    messages: Union[str, List[dict]] = None,
    model: Optional[str] = "gpt-4o-mini",
    response_model: Optional[BaseModel] = None,
    mode: Optional[ClientModeParams] = "md_json",
    optimize: Union[Optimizer, None] = None,
    tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
    run_tools: Optional[bool] = True,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    params: Optional[ClientParams] = None,
    verbose: Optional[bool] = False,
    stream: Optional[bool] = False,
    image_urls: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> Response:
    """
    Runs a tool calling & structured output capable completion or completion chain.
    
    Example:
        ```python
        import zyx
        zyx.completion()
        ```
        
    Args:
        messages (Union[str, List[dict]]): The messages to send to the model.
        model (Optional[str]): The model to use for completion.
        response_model (Optional[BaseModel]): The response model to use for structured output.
        mode (Optional[ClientModeParams]): The mode to use for completion.
        optimize (Union[Optimizer, None]): The optimization strategy to use for completion.
        tools (Optional[List[Union[Callable, dict, BaseModel]]]): The tools to use for completion.
        run_tools (Optional[bool]): Whether to run the tools.
        base_url (Optional[str]): The base URL to use for the API.
        api_key (Optional[str]): The API key to use for authentication.
        organization (Optional[str]): The organization to use for the API.
        top_p (Optional[float]): The top p value for sampling.
        temperature (Optional[float]): The temperature value for sampling.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
        max_retries (Optional[int]): The maximum number of retries for the API call.
        params (Optional[ClientParams]): Additional parameters for the completion.
        verbose (Optional[bool]): Whether to print verbose output.
        stream (Optional[bool]): Whether to stream the response.
        image_urls (Optional[Union[str, List[str]]]): The image URLs to interpret.
    
    Returns:
        Response: The completion response, or a generator if streaming is enabled.
    """
    import tenacity

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        retry=tenacity.retry_if_exception_type(Exception),
    )
    def completion_with_retry(client, current_mode):
        try:
            # Run Completion
            return client.completion(
                messages=messages,
                model=model,
                tools=tools,
                run_tools=run_tools,
                response_model=response_model,
                mode=current_mode,
                base_url=base_url,
                api_key=api_key,
                organization=organization,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                optimize=optimize,
                params=params,
                verbose=verbose,
                image_urls=image_urls,
                **kwargs,
            )

        except Exception as e:
            # Run Tenacity Retry if Instructor Mode is used
            if response_model:
                if current_mode == "md_json":
                    new_mode = "tools"
                else:
                    new_mode = "md_json"
                print(f"Retrying with mode: {new_mode}")
                return completion_with_retry(client, new_mode)

            # Raise Error if Instructor Mode is not used
            else:
                raise e

    try:
        client = CompletionClient()
        client_params = params or ClientParams(
            messages=client.format_messages(messages),
            model=model,
            tools=client.format_tools(tools),
            run_tools=run_tools,
            response_model=response_model,
            mode=mode,
            base_url=base_url,
            api_key=api_key,
            organization=organization,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            kwargs=kwargs,
        )
        client.params = client_params

        if optimize:
            client.params.messages = client.optimize_system_prompt(
                optimize=optimize, verbose=verbose
            )

        if image_urls:
            image_interpretation = client.interpret_images(image_urls, model)
            client.params.messages.append(
                {"role": "assistant", "content": image_interpretation}
            )

        if stream and not response_model:
            return client.stream_completion()
        else:
            return completion_with_retry(client, mode)

    except Exception as e:
        print(f"Error in completion function: {e}")
        return None


if __name__ == "__main__":
    print(completion("hi"))
