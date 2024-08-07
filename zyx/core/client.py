__all__ = [
    "completion",
    "embeddings",
    "zyxLanguageClient",
]

from functools import wraps
from .ext import BaseModel
from typing import Any, Callable, List, Literal, Optional, Union, get_type_hints


class zyxLanguageClientParams(BaseModel):
    model: Optional[str] = "openai/gpt-4o-mini"
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0
    max_retries: Optional[int] = 3
    verbose: Optional[bool] = False


class zyxLanguageClient:
    def __init__(
        self,
        model: Optional[str] = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = 3,
        temperature: Optional[float] = 0,
        verbose: Optional[bool] = False,
        params: Optional[zyxLanguageClientParams] = None,
    ):
        """
        A client for interacting with various language models.

        Args:
            model (str): The model to connect to. Defaults to "openai/gpt-4o-mini".
            api_key (str): The API key to use for authentication.
            base_url (str): The base URL to use for the API.
            organization (str): The organization to use for the API.
            max_tokens (int): The maximum number of tokens to generate in a single request.
            max_retries (int): The maximum number of retries to make before failing.
            temperature (float): The temperature to use for sampling.
            verbose (bool): Whether to show tool calls.
            params (zyxLanguageClientParams): The parameters to use for the client
        """
        if not params:
            self.params = zyxLanguageClientParams(
                model=model,
                provider=None,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                max_tokens=max_tokens,
                max_retries=max_retries,
                temperature=temperature,
                verbose=verbose,
            )
        else:
            self.params = params

        self.llm = self._connect_llm()
        if self.params.verbose:
            print(
                f"Connected to [bold green]{self.params.model}[/bold green] through the [bold blue]{self.params.provider}[/bold blue] client."
            )

    def _connect_llm(self):
        if (
            self.params.model.startswith("openai/")
            or self.params.model.startswith("gpt-")
            or self.params.base_url is not None
        ):
            if self.params.model.startswith("openai/"):
                self.params.model = self.params.model[7:]
            self.params.provider = "OpenAI"
            from phi.llm.openai.chat import OpenAIChat

            return OpenAIChat(
                model=self.params.model,
                max_tokens=self.params.max_tokens,
                temperature=self.params.temperature,
                organization=self.params.organization,
                api_key=self.params.api_key,
                base_url=self.params.base_url,
                max_retries=self.params.max_retries,
                show_tool_calls=self.params.verbose,
            )
        elif self.params.model.startswith("anthropic/") or self.params.model.startswith(
            "claude-"
        ):
            if self.params.model.startswith("anthropic/"):
                self.params.model = self.params.model[9:]
            from phi.llm.anthropic.claude import Claude

            self.params.provider = "Anthropic"
            return Claude(
                model=self.params.model,
                max_tokens=self.params.max_tokens,
                temperature=self.params.temperature,
                api_key=self.params.api_key,
                show_tool_calls=self.params.verbose,
            )
        elif self.params.model.startswith("ollama/") or self.params.model.startswith(
            "ollama_chat/"
        ):
            if self.params.model.startswith("ollama/"):
                self.params.model = self.params.model[7:]
            if self.params.model.startswith("ollama_chat/"):
                self.params.model = self.params.model[12:]
            self.params.provider = "Ollama"
            from phi.llm.ollama.chat import Ollama

            return Ollama(
                model=self.params.model,
                show_tool_calls=self.params.verbose,
            )
        elif self.params.model.startswith("mistral/"):
            self.params.provider = "Mistral"
            self.params.model = self.params.model[8:]
            from phi.llm.mistral.mistral import Mistral

            return Mistral(
                model=self.params.model,
                temperature=self.params.temperature,
                api_key=self.params.api_key,
                show_tool_calls=self.params.verbose,
                max_retries=self.params.max_retries,
                max_tokens=self.params.max_tokens,
            )
        elif self.params.model.startswith("groq/"):
            self.params.provider = "Groq"
            self.params.model = self.params.model[5:]
            from phi.llm.groq.groq import Groq

            return Groq(
                model=self.params.model,
                api_key=self.params.api_key,
                show_tool_calls=self.params.verbose,
                temperature=self.params.temperature,
                max_retries=self.params.max_retries,
                max_tokens=self.params.max_tokens,
            )
        elif self.params.model.startswith("google/"):
            self.params.provider = "Google"
            self.params.model = self.params.model[7:]
            from phi.llm.google.gemini import Gemini

            return Gemini(
                model=self.params.model,
                api_key=self.params.api_key,
                show_tool_calls=self.params.verbose,
            )
        elif self.params.model.startswith("cohere/"):
            self.params.provider = "Cohere"
            self.params.model = self.params.model[7:]
            from phi.llm.cohere.chat import CohereChat

            return CohereChat(
                model=self.params.model,
                api_key=self.params.api_key,
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
                show_tool_calls=self.params.verbose,
            )
        else:
            raise ValueError(f"Model {self.params.model} not supported")

    def _run_completion(
        self,
        messages: Union[str, list[dict]],
        tools: Optional[list] = None,
        response_model: Optional[BaseModel] = None,
    ):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        from phi.assistant.assistant import Assistant

        assistant = Assistant(
            llm=self.llm,
            tools=tools,
            output_model=response_model,
            show_tool_calls=self.params.verbose,
        )
        return assistant.run(messages=messages, stream=False)


CompletionTools = Literal["web", "calculator", "shell", "python"]


def completion(
    messages: Union[str, list[dict]],
    model: Optional[str] = "openai/gpt-4o-mini",
    tools: Optional[
        Union[CompletionTools, Callable, List[Union[CompletionTools, Callable]]]
    ] = None,
    response_model: Optional[BaseModel] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    temperature: Optional[float] = 0,
    verbose: Optional[bool] = False,
):
    """
    Entrypoint function for interacting with language models.

    Example:
        ```python
        import zyx

        response = zyx.completion("hi")
        print(response)
        ```

    Parameters:
        messages (Union[str, list[dict]]): The messages to send to the language model.
        model (str): The model to connect to. Defaults to "openai/gpt-4o-mini".
        tools (list): The tools to use for the completion.
        response_model (BaseModel): The response model to use for the completion.
        api_key (str): The API key to use for authentication.
        base_url (str): The base URL to use for the API.
        organization (str): The organization to use for the API.
        max_tokens (int): The maximum number of tokens to generate in a single request.
        max_retries (int): The maximum number of retries to make before failing.
        temperature (float): The temperature to use for sampling.
        verbose (bool): Whether to show tool calls.
    """
    client = zyxLanguageClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
        verbose=verbose,
    )
    if tools:
        if "web" in tools:
            from .toolkits import WebsiteTools, DuckDuckGo

            tools.append([WebsiteTools() + DuckDuckGo()])
        elif "calculator" in tools:
            from .toolkits import Calculator

            tools.append(Calculator())
        elif "shell" in tools:
            from .toolkits import ShellTools

            tools.append(ShellTools())
        elif "python" in tools:
            from .toolkits import PythonTools

            tools.append(PythonTools())
    return client._run_completion(
        messages=messages, tools=tools, response_model=response_model
    )


def function(
    model: Union[Callable, str] = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Creates an emulated response for the function below.
    
    Example:
        ```python
        import zyx
        
        @zyx.function
        def parse_doc(doc : str) -> str:
            """ "Parses the document and generates a 500 word summary." """
        
        response = parse_doc("./docs/doc_about_monkeys.txt)
        print(response)
        ```
    
    Parameters:
        model (Union[Callable, str]): The model to use for the function.
        api_key (str): The API key to use for authentication.
        base_url (str): The base URL to use for the API.
    """
    import tenacity

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        @tenacity.retry(
            stop=tenacity.stop_after_attempt(3),
            retry=tenacity.retry_if_exception_type(),
        )
        def function_wrapper(*args, **kwargs):
            from pydantic import create_model

            type_hints = get_type_hints(f)
            return_type = type_hints.pop("return", Any)
            FunctionResponseModel = create_model(
                "FunctionResponseModel",
                input=(dict[str, Any], ...),
                output=(return_type, ...),
            )
            function_args = {k: str(v) for k, v in type_hints.items()}
            input_dict = dict(zip(function_args.keys(), args))
            input_dict.update(kwargs)
            messages = [
                {
                    "role": "system",
                    "content": f"""
                ## CONTEXT ##
                You are a python function emulator. Your only goal is to simulate the response of a python function.
                You only respond in the format the python function would respond. You never chat, or use a conversational response.
                Function: {f.__name__}
                Arguments and their types: {function_args}
                Return type: {return_type}
                ## OBJECTIVE ##
                Plan out your reasoning before you begin to respond at all.
                Description: {f.__doc__}
                """,
                },
                {"role": "user", "content": f"INPUTS: {input_dict}"},
            ]
            response = completion(
                messages=messages,
                model=decorator.model,
                api_key=api_key,
                base_url=base_url,
                response_model=FunctionResponseModel,
                **kwargs,
            )
            return response.output

        return function_wrapper

    if callable(model):
        decorator.model = "openai/gpt-4o-mini"
        return decorator(model)
    else:
        decorator.model = model
        return decorator


PresetLanguages = Literal[
    "python",
    "javascript" "sql",
    "shell",
]


def code(
    instructions: str = None,
    language: Union[PresetLanguages, str] = "python",
    model: Optional[str] = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    temperature: Optional[float] = 0,
    tools: Optional[
        Union[CompletionTools, Callable, List[Union[CompletionTools, Callable]]]
    ] = None,
    verbose: Optional[bool] = False,
):
    system_prompt = f"""
    ## CONTEXT ##
    You are a code generator. Your only goal is provide code based on the instructions given.
    Language : {language}
    
    ## OBJECTIVE ##
    Plan out your reasoning before you begin to respond at all.
    """

    class CodeResponseModel(BaseModel):
        code: str

    response = completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instructions},
        ],
        model=model,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
        tools=tools,
        response_model=CodeResponseModel,
        verbose=verbose,
    )
    return response.code


def generate(
    target: BaseModel,
    instructions: Optional[str] = None,
    n: int = 1,
    model: Optional[str] = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    temperature: Optional[float] = 0,
    verbose: Optional[bool] = False,
):
    """
    Generates instances of a given Pydantic BaseModel based on instructions.

    Args:
        model (BaseModel): The Pydantic BaseModel to generate instances of.
        instructions (str, optional): Instructions for generating the model instances.
        n (int): Number of instances to generate. Defaults to 1.
        llm_model (str, optional): The language model to use. Defaults to "openai/gpt-4o-mini".
        api_key (str, optional): The API key for the language model.
        base_url (str, optional): The base URL for the API.
        organization (str, optional): The organization for the API.
        max_tokens (int, optional): Maximum number of tokens for the response.
        max_retries (int, optional): Maximum number of retries for the API call.
        temperature (float, optional): Temperature for response generation.
        verbose (bool, optional): Whether to show verbose output.

    Returns:
        List[BaseModel]: A list of generated model instances.
    """
    from pydantic import create_model

    ResponseModel = create_model("ResponseModel", items=(List[target], ...))

    # Prepare the system message
    system_message = f"""
    You are a data generator. Your task is to generate {n} valid instance(s) of the following Pydantic model:
    
    {target.schema_json(indent=2)}
    
    Ensure that all generated instances comply with the model's schema and constraints.
    """
    user_message = (
        instructions
        if instructions
        else f"Generate {n} instance(s) of the given model."
    )

    response = completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        model=model,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
        verbose=verbose,
        response_model=ResponseModel,
    )
    return response.items


def extract(
    target: BaseModel,
    text: str,
    model: Optional[str] = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    temperature: Optional[float] = 0,
    verbose: Optional[bool] = False,
):
    """
    Extracts information from the given text and fits it into the provided Pydantic BaseModel.

    Args:
        model (BaseModel): The Pydantic BaseModel to extract information into.
        text (str): The input text to extract information from.
        llm_model (str, optional): The language model to use. Defaults to "openai/gpt-4o-mini".
        api_key (str, optional): The API key for the language model.
        base_url (str, optional): The base URL for the API.
        organization (str, optional): The organization for the API.
        max_tokens (int, optional): Maximum number of tokens for the response.
        max_retries (int, optional): Maximum number of retries for the API call.
        temperature (float, optional): Temperature for response generation.
        verbose (bool, optional): Whether to show verbose output.

    Returns:
        BaseModel: An instance of the provided model with extracted information.
    """
    system_message = f"""
    You are an information extractor. Your task is to extract relevant information from the given text 
    and fit it into the following Pydantic model:
    
    {target}
    
    Only extract information that is explicitly stated in the text. Do not infer or generate any information 
    that is not present in the input text. If a required field cannot be filled with information from the text, 
    leave it as None or an empty string as appropriate.
    """

    user_message = f"Extract information from the following text and fit it into the given model:\n\n{text}"

    response = completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        model=model,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
        verbose=verbose,
        response_model=target,
    )

    return response


class ClassificationResult(BaseModel):
    text: str
    label: str


def classify(
    inputs: Union[str, List[str]],
    labels: List[str],
    n: int = 1,
    llm_model: Optional[str] = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    temperature: Optional[float] = 0,
    verbose: Optional[bool] = False,
):
    """
    Classifies the given input text(s) using the provided labels.

    Args:
        inputs (Union[str, List[str]]): The input text(s) to classify.
        labels (List[str]): The list of possible classification labels.
        n (int): Number of classifications to generate for each input. Defaults to 1.
        llm_model (str, optional): The language model to use. Defaults to "openai/gpt-4o-mini".
        api_key (str, optional): The API key for the language model.
        base_url (str, optional): The base URL for the API.
        organization (str, optional): The organization for the API.
        max_tokens (int, optional): Maximum number of tokens for the response.
        max_retries (int, optional): Maximum number of retries for the API call.
        temperature (float, optional): Temperature for response generation.
        verbose (bool, optional): Whether to show verbose output.

    Returns:
        Union[List[ClassificationResult], List[List[ClassificationResult]]]:
        A list of classification results, or a list of lists if multiple inputs are provided.
    """
    from pydantic import create_model

    ResponseModel = create_model(
        "ResponseModel", items=(List[ClassificationResult], ...)
    )

    system_message = f"""
    You are a text classifier. Your task is to classify the given text(s) into one of the following categories:
    {', '.join(labels)}
    
    For each input, provide {n} classification(s). Each classification should include the original text 
    and the assigned label.
    """

    if isinstance(inputs, str):
        inputs = [inputs]
    user_message = "Classify the following text(s):\n\n" + "\n\n".join(inputs)

    response = completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        model=llm_model,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
        verbose=verbose,
        response_model=ResponseModel,
    )

    results = response.items
    if len(inputs) == 1:
        return results
    else:
        grouped_results = []
        for i in range(0, len(results), n):
            grouped_results.append(results[i : i + n])
        return grouped_results
