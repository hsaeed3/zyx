# zyx ==============================================================================

__all__ = [
    "completion",
    "CompletionClient",
]

from typing import Literal, Optional, Union, Callable, List, Iterator
from zyx.client import BaseModel

CompletionTools = Literal["web", "calculator", "shell", "python"]
CompletionProviders = [
    "openai/",
    "anthropic/",
    "ollama/",
]


class CompletionClient:
    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",
        response_model: Optional[BaseModel] = None,
        name: Optional[str] = None,
        task: Optional[str] = None,
        introduction: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        prevent_hallucinations: Optional[bool] = False,
        prevent_prompt_injection: Optional[bool] = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[
            Union[CompletionTools, Callable, List[Union[CompletionTools, Callable]]]
        ] = None,
        verbose: bool = False,
        markdown: bool = True,
        *args,
        **kwargs,
    ):
        self.verbose = verbose
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.response_model = response_model
        self.tools = tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.organization = organization
        self.name = name
        self.introduction = introduction
        self.description = description
        self.instructions = instructions
        self.task = task
        self.prevent_hallucinations = prevent_hallucinations
        self.prevent_prompt_injection = prevent_prompt_injection
        self.args = args
        self.kwargs = kwargs
        self.markdown = markdown
        self.agent_tools = None

        if self.verbose is True:
            from zyx.client import logger

            self.logger = logger
        if self.verbose is True:
            self.logger.info(f"Initializing completion client with model: {model}")
        if self.tools:
            if self.verbose is True:
                logger.info(f"Initializing completion client with tools: {tools}")
            self._build_tools()

        self._build_completion_agent()

    def _build_tools(self):
        if isinstance(self.tools, list):
            self.agent_tools = []
            for tool in self.tools:
                if callable(tool):
                    self.agent_tools.append(tool)
                elif tool == "web":
                    from phi.tools.duckduckgo import DuckDuckGo
                    from phi.tools.website import WebsiteTools

                    self.agent_tools.append(WebsiteTools())
                    self.agent_tools.append(DuckDuckGo())
                elif tool == "calculator":
                    from phi.tools.calculator import Calculator

                    self.agent_tools.append(Calculator())
                elif tool == "shell":
                    from phi.tools.shell import ShellTools

                    self.agent_tools.append(ShellTools(base_dir="."))
                elif tool == "python":
                    from phi.tools.python import PythonTools

                    self.agent_tools.append(PythonTools(base_dir="."))
        elif callable(self.tools):
            self.agent_tools = self.tools

    def _build_completion_agent(self):
        from phi.assistant.assistant import Assistant

        if self.base_url is not None:
            if self.verbose is True:
                self.logger.info("Attempting to initialize custom provider.")
            try:
                from phi.llm.openai.chat import OpenAIChat

                if self.agent_tools is not None:
                    self.assistant = Assistant(
                        llm=OpenAIChat(
                            model=self.model,
                            base_url=self.base_url,
                            api_key=self.api_key,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            organization=self.organization,
                            show_tool_calls=self.verbose,
                        ),
                        name=self.name,
                        introduction=self.introduction,
                        description=self.description,
                        instructions=self.instructions,
                        task=self.task,
                        tools=self.agent_tools,
                        prevent_hallucinations=self.prevent_hallucinations,
                        prevent_prompt_injection=self.prevent_prompt_injection,
                        output_model=self.response_model,
                        markdown=self.markdown,
                        parse_output=False,
                        *self.args,
                        **self.kwargs,
                    )
                else:
                    self.assistant = Assistant(
                        llm=OpenAIChat(
                            model=self.model,
                            base_url=self.base_url,
                            api_key=self.api_key,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            organization=self.organization,
                            show_tool_calls=self.verbose,
                        ),
                        name=self.name,
                        introduction=self.introduction,
                        description=self.description,
                        instructions=self.instructions,
                        task=self.task,
                        prevent_hallucinations=self.prevent_hallucinations,
                        prevent_prompt_injection=self.prevent_prompt_injection,
                        output_model=self.response_model,
                        markdown=self.markdown,
                        parse_output=False,
                        *self.args,
                        **self.kwargs,
                    )
            except Exception as e:
                if self.verbose is True:
                    self.logger.error(f"Failed to initialize custom provider. {e}")
                else:
                    raise e
        elif self.model.startswith("openai/"):
            self.model = self.model.split("/")[1]
            try:
                from phi.llm.openai.chat import OpenAIChat

                if self.agent_tools is not None:
                    self.assistant = Assistant(
                        llm=OpenAIChat(
                            model=self.model,
                            base_url=self.base_url,
                            api_key=self.api_key,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            organization=self.organization,
                            show_tool_calls=self.verbose,
                        ),
                        name=self.name,
                        introduction=self.introduction,
                        description=self.description,
                        instructions=self.instructions,
                        task=self.task,
                        tools=self.agent_tools,
                        prevent_hallucinations=self.prevent_hallucinations,
                        prevent_prompt_injection=self.prevent_prompt_injection,
                        output_model=self.response_model,
                        markdown=self.markdown,
                        parse_output=False,
                        *self.args,
                        **self.kwargs,
                    )
                else:
                    self.assistant = Assistant(
                        llm=OpenAIChat(
                            model=self.model,
                            base_url=self.base_url,
                            api_key=self.api_key,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            organization=self.organization,
                            show_tool_calls=self.verbose,
                        ),
                        name=self.name,
                        introduction=self.introduction,
                        description=self.description,
                        instructions=self.instructions,
                        task=self.task,
                        prevent_hallucinations=self.prevent_hallucinations,
                        prevent_prompt_injection=self.prevent_prompt_injection,
                        output_model=self.response_model,
                        markdown=self.markdown,
                        parse_output=False,
                        *self.args,
                        **self.kwargs,
                    )
            except Exception as e:
                if self.verbose is True:
                    self.logger.error(f"Failed to initialize OpenAI provider. {e}")
                else:
                    raise e
        elif self.model.startswith("anthropic/"):
            self.model = self.model.split("/")[1]
            try:
                from phi.llm.anthropic.claude import Claude

                if self.agent_tools is not None:
                    self.assistant = Assistant(
                        llm=Claude(
                            model=self.model,
                            api_key=self.api_key,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            show_tool_calls=self.verbose,
                        ),
                        name=self.name,
                        introduction=self.introduction,
                        description=self.description,
                        instructions=self.instructions,
                        task=self.task,
                        tools=self.agent_tools,
                        prevent_hallucinations=self.prevent_hallucinations,
                        prevent_prompt_injection=self.prevent_prompt_injection,
                        output_model=self.response_model,
                        markdown=self.markdown,
                        parse_output=False,
                        *self.args,
                        **self.kwargs,
                    )
                else:
                    self.assistant = Assistant(
                        llm=Claude(
                            model=self.model,
                            api_key=self.api_key,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            show_tool_calls=self.verbose,
                        ),
                        name=self.name,
                        introduction=self.introduction,
                        description=self.description,
                        instructions=self.instructions,
                        task=self.task,
                        prevent_hallucinations=self.prevent_hallucinations,
                        prevent_prompt_injection=self.prevent_prompt_injection,
                        output_model=self.response_model,
                        markdown=self.markdown,
                        parse_output=False,
                        *self.args,
                        **self.kwargs,
                    )
            except Exception as e:
                if self.verbose is True:
                    self.logger.error(f"Failed to initialize Anthropic provider. {e}")
                else:
                    raise e
        elif self.model.startswith("ollama/"):
            self.model = self.model.split("/")[1]
            try:
                from phi.llm.ollama.openai import OllamaOpenAI

                if self.agent_tools is not None:
                    self.assistant = Assistant(
                        llm=OllamaOpenAI(
                            model=self.model,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            show_tool_calls=self.verbose,
                        ),
                        name=self.name,
                        introduction=self.introduction,
                        description=self.description,
                        instructions=self.instructions,
                        task=self.task,
                        tools=self.agent_tools if self.agent_tools else None,
                        prevent_hallucinations=self.prevent_hallucinations,
                        prevent_prompt_injection=self.prevent_prompt_injection,
                        output_model=self.response_model,
                        markdown=self.markdown,
                        parse_output=False,
                        *self.args,
                        **self.kwargs,
                    )
                else:
                    self.assistant = Assistant(
                        llm=OllamaOpenAI(
                            model=self.model,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            show_tool_calls=self.verbose,
                        ),
                        name=self.name,
                        introduction=self.introduction,
                        description=self.description,
                        instructions=self.instructions,
                        task=self.task,
                        prevent_hallucinations=self.prevent_hallucinations,
                        prevent_prompt_injection=self.prevent_prompt_injection,
                        output_model=self.response_model,
                        markdown=self.markdown,
                        parse_output=False,
                        *self.args,
                        **self.kwargs,
                    )
            except Exception as e:
                if self.verbose is True:
                    self.logger.error(f"Failed to initialize Ollama provider. {e}")
                else:
                    raise e
        else:
            raise ValueError(
                f"Invalid model or provider, please use the litellm format ('openai/gpt-3.5-turbo'): {self.model}"
            )


# --- completion -------------------------------------------------------------------


def completion(
    messages: Union[str, list[str]] = None,
    model: str = "openai/gpt-3.5-turbo",
    response_model: Optional[BaseModel] = None,
    name: Optional[str] = None,
    task: Optional[str] = None,
    introduction: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    prevent_hallucinations: Optional[bool] = False,
    prevent_prompt_injection: Optional[bool] = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tools: Optional[
        Union[CompletionTools, Callable, List[Union[CompletionTools, Callable]]]
    ] = None,
    verbose: bool = False,
    markdown: bool = False,
    stream: bool = False,
    chat: Optional[bool] = False,
    *args,
    **kwargs,
) -> Union[Iterator[str], str, BaseModel]:
    """A function built on top of PhiData's abstractions to provide a quick and unified chat completions
    client with both tool calling & structured output.

    Example:
        ```python
        import zyx

        zyx.completion("How are you today?")
        ```

    Args:
        messages (Union[str, list[str], optional): The message or messages to send to the completion model.
        model (str, optional): The completion model to use. Defaults to "openai/gpt-3.5-turbo".
        response_model (Optional[BaseModel], optional): The response model to use. Defaults to None.

        name (Optional[str], optional): The name of the completion agent. Defaults to None.
        task (Optional[str], optional): The task of the completion agent. Defaults to None.
        introduction (Optional[str], optional): The introduction of the completion agent. Defaults to None.
        description (Optional[str], optional): The description of the completion agent. Defaults to None.
        instructions (Optional[str], optional): The instructions of the completion agent. Defaults to None.
        system_prompt (Optional[str], optional): The system prompt of the completion agent. Defaults to None.
        prevent_hallucinations (Optional[bool], optional): Prevent hallucinations in the completion agent. Defaults to None.
        prevent_prompt_injection (Optional[bool], optional): Prevent prompt injection in the completion agent. Defaults to None.

        base_url (Optional[str], optional): The base url of the completion agent. Defaults to None.
        api_key (Optional[str], optional): The api key of the completion agent. Defaults to None.
        organization (Optional[str], optional): The organization of the completion agent. Defaults to None.
        max_tokens (Optional[int], optional): The max tokens of the completion agent. Defaults to None.
        temperature (Optional[float], optional): The temperature of the completion agent. Defaults to None.

        tools (Optional[Union[CompletionTools, Callable, List[Union[CompletionTools, Callable]]]], optional): The tools to use in the completion agent. Defaults to None.
        verbose (bool, optional): Whether to show verbose output. Defaults to False.
        markdown (bool, optional): Whether to use markdown in the completion agent. Defaults to True.
        stream (bool, optional): Whether to stream the completion agent. Defaults to False.

        chat (Optional[bool], optional): Start Chat in CLI Mode.
    """
    if messages is None:
        raise ValueError(
            "Please provide a message or messages to send to the completion model."
        )
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    try:
        assistant_client = CompletionClient(
            model=model,
            response_model=response_model,
            name=name,
            task=task,
            introduction=introduction,
            description=description,
            instructions=instructions,
            prevent_hallucinations=prevent_hallucinations,
            prevent_prompt_injection=prevent_prompt_injection,
            base_url=base_url,
            api_key=api_key,
            organization=organization,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            verbose=verbose,
            markdown=markdown,
            stream=stream,
            *args,
            **kwargs,
        )
    except Exception as e:
        raise e

    if chat is True:
        assistant_client.assistant.cli_app(markdown=markdown, exit_on=["exit", "quit"])
    else:
        try:
            response = assistant_client.assistant.run(
                messages=messages,
                stream=stream,
            )
            return response
        except Exception as e:
            raise e
