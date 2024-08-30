__all__ = ["Client", "completion"]

# --- zyx ----------------------------------------------------------------

from ..core.types import ClientParams, ClientModeParams, ClientToolParams
from ..core.utils.convert_to_openai_tool import convert_to_openai_tool
from ..core.main import BaseModel
from typing import Callable, List, Optional, Union, Generator
from litellm.types.utils import ModelResponse


class Client:
    def __init__(
        self,
    ):
        pass

    @staticmethod
    def format_messages(messages: Union[str, list[dict]] = None) -> list[dict]:
        try:
            if isinstance(messages, str):
                return [{"role": "user", "content": messages}]
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
        try:
            if not tools:
                return None
            params = ClientToolParams(tools=tools, openai_tools=[], mapping={})
            for tool in tools:
                if isinstance(tool, dict):
                    params.openai_tools.append(tool)
                elif isinstance(tool, BaseModel) or isinstance(tool, Callable):
                    params.openai_tools.append(convert_to_openai_tool(tool))
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
        try:
            from litellm.main import completion

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

    def completion(
        self,
        messages: Union[str, list[dict]] = None,
        model: Optional[str] = "gpt-4o-mini",
        tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
        run_tools: Optional[bool] = True,
        response_model: Optional[BaseModel] = None,
        mode: Optional[ClientModeParams] = "md_json",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = 3,
        params: Optional[ClientParams] = None,
        verbose: Optional[bool] = False,
        **kwargs,
    ):
        """Runs a tool calling & structured output capable completion or completion chain.

        Example:
            ```python

            import zyx

            zyx.completion()
            ```

        Parameters:
            - messages (Union[str, list[dict]]): The messages to send to the model.
            - model (Optional[str]): The model to use for completion.
            - tools (Optional[List[Union[Callable, dict, BaseModel]]]): The tools to use for completion.
            - run_tools (Optional[bool]): Whether to run the tools.
            - response_model (Optional[BaseModel]): The response model to use for completion.
            - mode (Optional[ClientModeParams]): The mode to use for completion.
            - base_url (Optional[str]): The base URL to use for completion.
            - api_key (Optional[str]): The API key to use for completion.
            - organization (Optional[str]): The organization to use for completion.
            - top_p (Optional[float]): The top p value to use for completion.
            - temperature (Optional[float]): The temperature value to use for completion.
            - max_tokens (Optional[int]): The maximum tokens to use for completion.
            - max_retries (Optional[int]): The maximum retries to use for completion.
            - params (Optional[ClientParams]): The parameters to use for completion.
            - verbose (Optional[bool]): Whether to print the parameters.

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
                        self.params.messages = self.execute_tools(message.tool_calls)
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
    tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
    run_tools: Optional[bool] = True,
    response_model: Optional[BaseModel] = None,
    mode: Optional[ClientModeParams] = "md_json",
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
    **kwargs,
) -> Union[ModelResponse, BaseModel, Generator]:
    """Runs a tool calling & structured output capable completion or completion chain.
    Example:
    ```python
    import zyx
    zyx.completion()
    ```
    Parameters:
    - messages (Union[str, list[dict]]): The messages to send to the model.
    - model (Optional[str]): The model to use for completion.
    - tools (Optional[List[Union[Callable, dict, BaseModel]]]): The tools to use for completion.
    - run_tools (Optional[bool]): Whether to run the tools.
    - response_model (Optional[BaseModel]): The response model to use for completion.
    - mode (Optional[ClientModeParams]): The mode to use for completion.
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
    Returns:
    - The completion response, or a generator if streaming.
    """
    import tenacity

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        retry=tenacity.retry_if_exception_type(Exception),
    )
    def completion_with_retry(client, current_mode):
        try:
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
                params=params,
                verbose=verbose,
                **kwargs,
            )
        except Exception as e:
            if response_model:
                # Switch mode for retry
                new_mode = "tools" if current_mode == "md_json" else "md_json"
                print(f"Retrying with mode: {new_mode}")
                return completion_with_retry(client, new_mode)
            else:
                raise e

    try:
        client = Client()
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

        if stream and not response_model:
            return client.stream_completion()
        else:
            return completion_with_retry(client, mode)

    except Exception as e:
        print(f"Error in completion function: {e}")
        return None


if __name__ == "__main__":
    print(completion("hi"))
