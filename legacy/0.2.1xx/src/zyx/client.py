# zyx ==============================================================================

__all__ = [
    "completion",
    "CompletionClient",
    "embeddings" "_cast",
    "_extract",
    "_classify",
    "_function",
    "_generate",
    "_inference",
]

from typing import Literal, Optional, Union, Callable, List, Iterator, Type
from .core import BaseModel, _UtilLazyLoader

CompletionTools = Literal["web", "calculator", "shell", "python"]
EmbeddingsProviders = ["google/", "openai/", "ollama/"]
CompletionProviders = [
    "openai/",
    "anthropic/",
    "ollama/",
]
PresetTools = Literal["web", "calculator", "shell", "python"]
ModelResponse = Type["ModelResponse"]
Assistant = Type["Assistant"]


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
            from zyx import logger

            self.logger = logger
        if self.verbose is True:
            self.logger.info(f"Initializing completion client with model: {model}")
        if self.tools is None:
            response = instructor_completion(
                messages=self.messages,
                model=self.model,
                response_model=self.response_model,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                debug=self.verbose,
                **self.kwargs,
            )
            self._return_response(response)
        elif self.tools is not None:
            if self.verbose is True:
                logger.info(f"Initializing completion client with tools: {tools}")
            self._build_tools()

        self._build_completion_agent()

    @staticmethod
    def _return_response(response):
        return response

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
                from phi.llm.ollama.chat import Ollama

                if self.agent_tools is not None:
                    self.assistant = Assistant(
                        llm=Ollama(
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
                        llm=Ollama(
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


# ==================================================================================


def embeddings(
    inputs: Union[list[str], str],
    model: Optional[str] = "openai/text-embedding-ada-002",
    dimensions: Optional[int] = None,
    host: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    encoding_format: Literal["float", "base64"] = "float",
    verbose: Optional[bool] = False,
    *args,
    **kwargs,
):
    """Utilize an external service to retrieve embeddings for the inputs text.

    Args:
        inputs (Union[list[str], str]): The inputs text to embed.
        model (Optional[str], optional): The model to use for embeddings. Defaults to "openai/text-embedding-ada-002".
        dimensions (Optional[int], optional): The number of dimensions for the embeddings. Defaults to None.
        host (Optional[str], optional): The host for the Ollama service. Defaults to None.
        api_key (Optional[str], optional): The API key for the service. Defaults to None.
        base_url (Optional[str], optional): The base URL for the service. Defaults to None.
        organization (Optional[str], optional): The organization for the service. Defaults to None.
        encoding_format (Literal["float", "base64"], optional): The encoding format for the embeddings. Defaults to "float".
        verbose (Optional[bool], optional): Whether to log information. Defaults to False.
    """
    if not inputs:
        raise ValueError("Inputs text is required.")
    if verbose is True:
        from zyx.core import logger

    if not any([model.startswith(provider) for provider in EmbeddingsProviders]):
        from phi.embedder.openai import OpenAIEmbedder

        try:
            embedder = OpenAIEmbedder(
                dimensions=dimensions,
                model=model,
                encoding_format=encoding_format,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
            )

            if isinstance(inputs, List):
                embeddings = []
                for i in inputs:
                    result = embedder.get_embedding(i)
                    embeddings.append(result)
                return embeddings
            else:
                return embedder.get_embedding(inputs)
        except Exception as e:
            if verbose is True:
                logger.error(f"Failed to initialize OpenAI Embedder: {e}")
            raise e
    elif model.startswith("openai/"):
        from phi.embedder.openai import OpenAIEmbedder

        model = model[7:]
        try:
            embedder = OpenAIEmbedder(
                dimensions=dimensions if dimensions is not None else 1536,
                model=model,
                encoding_format=encoding_format,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
            )
            if isinstance(inputs, List):
                embeddings = []
                for i in inputs:
                    result = embedder.get_embedding(i)
                    embeddings.append(result)
                return embeddings
            else:
                return embedder.get_embedding(inputs)
        except Exception as e:
            if verbose is True:
                logger.error(f"Failed to initialize OpenAI Embedder: {e}")
            raise e
    elif model.startswith("google/"):
        model = model[8:]
        from phi.embedder.google import GeminiEmbedder

        try:
            embedder = GeminiEmbedder(
                dimensions=dimensions if dimensions is not None else 1536,
                model=model,
                api_key=api_key,
            )
            if isinstance(inputs, List):
                embeddings = []
                for i in inputs:
                    result = embedder.get_embedding(i)
                    embeddings.append(result)
                return embeddings
            else:
                return embedder.get_embedding(inputs)
        except Exception as e:
            if verbose is True:
                logger.error(f"Failed to initialize Google Embedder: {e}")
            raise e
    elif model.startswith("ollama/"):
        model = model[8:]
        from phi.embedder.ollama import OllamaEmbedder

        try:
            embedder = OllamaEmbedder(
                dimensions=dimensions if dimensions is not None else 4096,
                model=model,
                api_key=api_key,
                host=host,
            )
            if isinstance(inputs, List):
                embeddings = []
                for i in inputs:
                    result = embedder.get_embedding(i)
                    embeddings.append(result)
                return embeddings
            else:
                return embedder.get_embedding(inputs)
        except Exception as e:
            if verbose is True:
                logger.error(f"Failed to initialize Ollama Embedder: {e}")
            raise e


# ==============================================================================


def instructor_completion(
    messages: Union[str, list[str]],
    model: Optional[str] = "openai/gpt-3.5-turbo",
    response_model: Type["BaseModel"] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = 0.5,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    strict: Optional[bool] = True,
    debug: Optional[bool] = False,
    **kwargs,
) -> Union["ModelResponse", "BaseModel"]:
    """
    Runs a litellm completion, tied in with the Instructor framework.

    Parameters:
        messages (Union[str, list[str]]) : The message(s) to send to the model
        model (str) : The model to use for completion
        response_model (Type['BaseModel']) : The response model to use
        base_url (Optional[str]) : The base url for the instructor
        temperature (Optional[float]) : The temperature for the completion
        max_tokens (Optional[int]) : The maximum tokens to use for the completion
        max_retries (Optional[int]) : The maximum retries to use for the completion
        strict (bool) : Whether to use strict mode for the completion

    Returns:
        Union['ModelResponse', 'BaseModel'] : The response from the completion
    """

    if not messages:
        raise ValueError("No messages provided")

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if response_model is None:
        from litellm.main import completion as litellm_completion

        response = litellm_completion(
            model=model,
            messages=messages,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **kwargs,
        )
        return _return_response(response)

    if model.startswith("ollama/"):
        return _ollama_instruct(
            model,
            messages,
            response_model,
            temperature=temperature,
            base_url=base_url,
            max_tokens=max_tokens,
            max_retries=max_retries,
            strict=strict,
            **kwargs,
        )
    else:
        return _litellm_instruct(
            model,
            messages,
            response_model,
            temperature=temperature,
            base_url=base_url,
            max_tokens=max_tokens,
            max_retries=max_retries,
            strict=strict,
            **kwargs,
        )


# --- util ----------------------------------------------------------------------


def _return_response(response: "ModelResponse" = None) -> "ModelResponse":
    """
    Returns the response from the completion function

    Parameters:
        response : The response from the completion function
    """
    return response


def _ollama_instruct(
    model: str,
    messages: list[dict],
    response_model: Type["BaseModel"],
    base_url: Optional[str] = None,
    temperature: Optional[float] = 0.5,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    strict: bool = True,
    **kwargs,
):
    """
    Runs the ollama instructor function

    Parameters:
        model (str) : The model to use for completion
        messages (list[dict]) : A list of messages to send to the model
        response_model (Type['BaseModel']) : The response model to use
        temperature (float) : The temperature to use for completion
        max_tokens (int) : The maximum number of tokens to generate
        max_retries (int) : The maximum number of retries to attempt
        strict (bool) : Whether to raise an exception on failure
        **kwargs : Additional arguments to pass to the completion function
    """
    if not base_url:
        base_url = "http://localhost:11434/v1"

    from instructor.client import from_openai as FromOpenAI
    from instructor.mode import Mode
    from openai import OpenAI

    client = FromOpenAI.from_openai(
        OpenAI(
            base_url=base_url,
            api_key="ollama",
        ),
        mode=Mode.JSON,
    )
    resp = client.chat.completions.create(
        model=model.split("/")[1],  # Remove the Ollama prefix
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        strict=strict,
        **kwargs,
    )
    return resp


def _litellm_instruct(
    model: str,
    messages: list[dict],
    response_model: Type["BaseModel"],
    base_url: Optional[str] = None,
    temperature: Optional[float] = 0.5,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    strict: bool = True,
    **kwargs,
):
    """
    Runs the litellm instructor function

    Parameters:
        model (str) : The model to use for completion
        messages (list[dict]) : A list of messages to send to the model
        response_model (Type['BaseModel']) : The response model to use
        temperature (float) : The temperature to use for completion
        max_tokens (int) : The maximum number of tokens to generate
        max_retries (int) : The maximum number of retries to attempt
        strict (bool) : Whether to raise an exception on failure
        **kwargs : Additional arguments to pass to the completion function
    """
    from litellm.main import completion as litellm_completion
    from instructor.client import from_litellm

    client = from_litellm(litellm_completion)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        base_url=base_url,
        max_tokens=max_tokens,
        max_retries=max_retries,
        strict=strict,
        **kwargs,
    )
    return response


# ==================================================================================


class _completion(_UtilLazyLoader):
    pass


_completion.init("zyx.client", "completion")


class _embeddings(_UtilLazyLoader):
    pass


_embeddings.init("zyx.client", "embeddings")


class _cast(_UtilLazyLoader):
    pass


_cast.init("marvin.ai.text", "cast")


class _extract(_UtilLazyLoader):
    pass


_extract.init("marvin.ai.text", "extract")


class _classify(_UtilLazyLoader):
    pass


_classify.init("marvin.ai.text", "classify")


class _function(_UtilLazyLoader):
    pass


_function.init("marvin.ai.text", "fn")


class _generate(_UtilLazyLoader):
    pass


_generate.init("marvin.ai.text", "generate")


class _inference(_UtilLazyLoader):
    pass


_inference.init("huggingface_hub.inference._client", "InferenceClient")
