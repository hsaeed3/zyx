__all__ = [
    "function",
    "chainofthought",
    "code",
    "generate",
    "extract",
    "classify",
]

# --- zyx ----------------------------------------------------------------


def chainofthought(
    query: str,
    answer_type: Any = str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0,
    verbose: bool = False,
):
    """
    Generates a chain of thought reasoning and extracts a final answer for a given query.

    Args:
        query (str): The question or problem to solve.
        answer_type (Any): The expected type of the answer (e.g., str, int, float).
        model (str): The model to use for completion.
        api_key (Optional[str]): The API key for authentication.
        base_url (Optional[str]): The base URL for the API.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
        temperature (float): The temperature for the model's output.
        verbose (bool): Whether to print verbose output.

    Returns:
        Any: The final answer of the specified type.
    """

    class Reasoning(BaseModel):
        chain_of_thought: str

    class FinalAnswer(BaseModel):
        answer: Any

    reasoning_prompt = f"""
    Let's approach this step-by-step:
    1. Understand the problem
    2. Identify key variables and their values
    3. Develop a plan to solve the problem
    4. Execute the plan, showing all calculations
    5. Verify the answer

    Question: {query}

    Now, let's begin our reasoning:
    """

    from .main import Client

    reasoning_response = Client().completion(
        messages=reasoning_prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        response_model=Reasoning,
        verbose=verbose,
    )

    if verbose:
        print("Chain of Thought:")
        print(reasoning_response.chain_of_thought)

    # Step 2: Extract final answer
    extraction_prompt = f"""
    Based on the following reasoning:
    {reasoning_response.chain_of_thought}

    Provide the final answer to the question: "{query}"
    Your answer should be of type: {answer_type.__name__}
    Only provide the final answer, without any additional explanation.
    """

    FinalAnswerModel = FinalAnswer[answer_type]  # type: ignore

    final_answer_response = Client().completion(
        messages=extraction_prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        response_model=FinalAnswerModel,
        verbose=verbose,
    )

    return final_answer_response.answer


# --- zyx ----------------------------------------------------------------

from ..core.ext import BaseModel
from typing import Any, Callable, Optional, get_type_hints, Union, List, Literal


def function(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> Callable[[Callable], Callable]:
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
        model (str): The model to use for the function.
        api_key (str): The API key to use for authentication.
        base_url (str): The base URL to use for the API.
    """
    import tenacity

    def decorator(f: Callable) -> Callable:
        from .main import Client
        from functools import wraps

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
            response = Client().completion(
                messages=messages,
                model=model,
                api_key=api_key,
                base_url=base_url,
                response_model=FunctionResponseModel,
                **kwargs,
            )
            return response.output

        return function_wrapper

    return decorator


# --- zyx ----------------------------------------------------------------


def code(
    instructions: str = None,
    language: Union[
        Literal[
            "python",
            "javascript",
            "typescript",
            "shell",
            "bash",
            "java",
            "cpp",
            "c++",
            "go",
            "sql",
        ],
        str,
    ] = "python",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0,
    tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
    verbose: bool = False,
):
    from .main import Client

    system_prompt = f"""
    ## CONTEXT ##
    You are a code generator. Your only goal is provide code based on the instructions given.
    Language : {language}
    
    ## OBJECTIVE ##
    Plan out your reasoning before you begin to respond at all.
    """

    class CodeResponseModel(BaseModel):
        code: str

    response = Client().completion(
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


# --- zyx ----------------------------------------------------------------


def generate(
    target: BaseModel,
    instructions: Optional[str] = None,
    n: int = 1,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0,
    verbose: bool = False,
):
    """
    Generates instances of a given Pydantic BaseModel based on instructions.

    Args:
        target (BaseModel): The Pydantic BaseModel to generate instances of.
        instructions (str, optional): Instructions for generating the model instances.
        n (int): Number of instances to generate. Defaults to 1.
        model (str, optional): The language model to use. Defaults to "gpt-4o-mini".
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
    from .main import Client
    from pydantic import create_model

    ResponseModel = create_model("ResponseModel", items=(List[target], ...))

    system_message = f"""
    You are a data generator. Your task is to generate {n} valid instance(s) of the following Pydantic model:
    
    {target.model_json_schema(indent=2)}
    
    Ensure that all generated instances comply with the model's schema and constraints.
    """
    user_message = (
        instructions
        if instructions
        else f"Generate {n} instance(s) of the given model."
    )

    response = Client().completion(
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


# --- zyx ----------------------------------------------------------------


def extract(
    target: BaseModel,
    text: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0,
    verbose: bool = False,
):
    """
    Extracts information from the given text and fits it into the provided Pydantic BaseModel.

    Args:
        target (BaseModel): The Pydantic BaseModel to extract information into.
        text (str): The input text to extract information from.
        model (str, optional): The language model to use. Defaults to "gpt-4o-mini".
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
    from .main import Client

    system_message = f"""
    You are an information extractor. Your task is to extract relevant information from the given text 
    and fit it into the following Pydantic model:
    
    {target}
    
    Only extract information that is explicitly stated in the text. Do not infer or generate any information 
    that is not present in the input text. If a required field cannot be filled with information from the text, 
    leave it as None or an empty string as appropriate.
    """

    user_message = f"Extract information from the following text and fit it into the given model:\n\n{text}"

    response = Client().completion(
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


# --- zyx ----------------------------------------------------------------


def classify(
    inputs: Union[str, List[str]],
    labels: List[str],
    n: int = 1,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0,
    verbose: bool = False,
):
    """
    Classifies the given input text(s) using the provided labels.

    Args:
        inputs (Union[str, List[str]]): The input text(s) to classify.
        labels (List[str]): The list of possible classification labels.
        n (int): Number of classifications to generate for each input. Defaults to 1.
        model (str, optional): The language model to use. Defaults to "gpt-4o-mini".
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
    from .main import Client

    class ClassificationResult(BaseModel):
        text: str
        label: str

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

    response = Client().completion(
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

    results = response.items
    if len(inputs) == 1:
        return results
    else:
        grouped_results = []
        for i in range(0, len(results), n):
            grouped_results.append(results[i : i + n])
        return grouped_results
