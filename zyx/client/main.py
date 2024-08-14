__all__ = [
    "Client", 
    "completion"
]

# --- zyx ----------------------------------------------------------------

from ..types import (
    ClientParams,
    ClientModeParams,
    ClientToolParams
)
from ..utils.tool_calling import convert_to_openai_tool
from ..core.ext import BaseModel
from typing import Callable, List, Optional, Union

class Client:
    def __init__(
        self,
    ):
        pass
    
    @staticmethod
    def format_messages(
        messages : Union[str, list[dict]] = None
    ) -> list[dict]:
        if isinstance(messages, str):
            return [{"role" : "user", "content" : messages}]
        return messages
    
    @staticmethod
    def format_tools(
        tools : List[Union[Callable, dict, BaseModel]] = None
    ):
        if not tools:
            return None
        params = ClientToolParams(tools = tools, openai_tools = [], mapping = {})
        for tool in tools:
            if isinstance(tool, dict):
                params.openai_tools.append(tool)
            elif isinstance(tool, BaseModel) \
                or isinstance(tool, Callable):
                params.openai_tools.append(convert_to_openai_tool(tool))
        for tool in tools:
            if isinstance(tool, Callable) or \
                isinstance(tool, BaseModel):
                params.mapping[tool.__name__] = tool
        return params
    
    def run_base_completion(self):
        from litellm.main import completion
        return completion(
            model = self.params.model,
            messages = self.params.messages,
            base_url = self.params.base_url,
            tools = self.params.tools.openai_tools if self.params.tools else None,
            api_key = self.params.api_key,
            organization = self.params.organization,
            top_p = self.params.top_p,
            temperature = self.params.temperature,
            max_tokens = self.params.max_tokens,
            **self.params.kwargs
        )
        
    def run_instructor_completion(self):
        from litellm.main import completion
        from instructor import from_litellm, Mode
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
        client = from_litellm(
            completion, mode = mode
        )
        return client.chat.completions.create(
            messages = self.params.messages,
            model = self.params.model,
            response_model = self.params.response_model,
            base_url = self.params.base_url,
            api_key = self.params.api_key,
            tools = self.params.tools.openai_tools if self.params.tools else None,
            organization = self.params.organization,
            top_p = self.params.top_p,
            temperature = self.params.temperature,
            max_tokens = self.params.max_tokens,
            max_retries = self.params.max_retries,
            **self.params.kwargs
        )
        
    def execute_tools(
        self,
        tool_calls : list[dict]
    ):
        import json
        
        if len(tool_calls) == 1:
            function_name = tool_calls[0].function.name
            function_args = json.loads(tool_calls[0].function.arguments)
            tool_result = self.params.tools.mapping[function_name](**function_args)
            self.params.messages.append({
                "tool_call_id" : tool_calls[0].id,
                "role" : "tool",
                "name" : function_name,
                "content" : str(tool_result)
            })
        else:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                tool_result = self.params.tools.mapping[function_name](**function_args)
                self.params.messages.append({
                    "tool_call_id" : tool_call.id,
                    "role" : "tool",
                    "name" : function_name,
                    "content" : str(tool_result)
                })
        return self.params.messages
        
    def completion(
        self,
        messages : Union[str, list[dict]] = None,
        
        model : Optional[str] = "gpt-4o-mini",
        tools : Optional[List[Union[Callable, dict, BaseModel]]] = None,
        run_tools : Optional[bool] = True,
        response_model : Optional[BaseModel] = None,
        mode : Optional[ClientModeParams] = "tools",
        
        base_url : Optional[str] = None,
        api_key : Optional[str] = None,
        organization : Optional[str] = None,
        
        top_p : Optional[float] = None,
        temperature : Optional[float] = None,
        max_tokens : Optional[int] = None,
        max_retries : Optional[int] = 3,
        params : Optional[ClientParams] = None,
        verbose : Optional[bool] = False,
        **kwargs
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
        if params:
            self.params = params
        else:
            self.params = ClientParams(
                messages = self.format_messages(messages),
                model = model,
                tools = self.format_tools(tools),
                run_tools = run_tools,
                response_model = response_model,
                mode = mode,
                base_url = base_url,
                api_key = api_key,
                organization = organization,
                top_p = top_p,
                temperature = temperature,
                max_tokens = max_tokens,
                max_retries = max_retries,
                kwargs = kwargs
            )
        if verbose:
            print(self.params + "\n")
            
        if not self.params.tools or not self.params.run_tools:
            if not self.params.response_model:
                return self.run_base_completion()
            else:
                return self.run_instructor_completion()
        response = self.run_base_completion()
        
        if not self.params.response_model:
            if not response.choices[0].message.tool_calls:
                return response
            else:
                while response.choices[0].message.tool_calls:
                    self.params.messages.append(response.choices[0].message)
                    self.params.messages = self.execute_tools(response.choices[0].message.tool_calls)
                    response = self.run_base_completion()
                return response
        else:
            if not response.choices[0].message.tool_calls:
                self.params.messages.append(response.choices[0].message)
                self.params.messages.append({"role" : "user", "content" : "Append tool responses to BaseModel."})
                self.params.tools = None
                return self.run_instructor_completion()
            else:
                while response.choices[0].message.tool_calls:
                    self.params.messages.append(response.choices[0].message)
                    self.params.messages = self.execute_tools(response.choices[0].message.tool_calls)
                    response = self.run_base_completion()
                self.params.messages.append({"role" : "user", "content" : "Append tool responses to BaseModel."})
                self.params.tools = None
                return self.run_instructor_completion()
            
def completion(
    messages : Union[str, list[dict]] = None,
    
    model : Optional[str] = "gpt-4o-mini",
    tools : Optional[List[Union[Callable, dict, BaseModel]]] = None,
    run_tools : Optional[bool] = True,
    response_model : Optional[BaseModel] = None,
    mode : Optional[ClientModeParams] = "tools",
    
    base_url : Optional[str] = None,
    api_key : Optional[str] = None,
    organization : Optional[str] = None,
    
    top_p : Optional[float] = None,
    temperature : Optional[float] = None,
    max_tokens : Optional[int] = None,
    max_retries : Optional[int] = 3,
    params : Optional[ClientParams] = None,
    verbose : Optional[bool] = False,
    **kwargs
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
    return Client().completion(
        messages = messages,
        model = model,
        tools = tools,
        run_tools = run_tools,
        response_model = response_model,
        mode = mode,
        base_url = base_url,
        api_key = api_key,
        organization = organization,
        top_p = top_p,
        temperature = temperature,
        max_tokens = max_tokens,
        max_retries = max_retries,
        params = params,
        verbose = verbose,
        **kwargs
    )   
        

                        
            
        
    