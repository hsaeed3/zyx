__all__ = ["code"]

from ...core.main import BaseModel
from typing import Optional, List, Union, Callable, Literal


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
    mode: Literal["json", "md_json", "tools"] = "md_json",
) -> str:
    """Generates code based on the instructions given.

    Example:
        ```python
        import zyx

        zyx.code(instructions="Print 'Hello, World!'")
        ```

    Parameters:
        instructions (str): The instructions to generate code for.
        language (Union[Literal[python, javascript, typescript, shell, bash, java, cpp, c++, go, sql], str]): The language to generate code for.
        model (str): The model to use for the code generation.
        api_key (Optional[str]): The API key to use for the code generation.
        base_url (Optional[str]): The base URL to use for the code generation.
        organization (Optional[str]): The organization to use for the code generation.
        max_tokens (Optional[int]): The maximum number of tokens to use for the code generation.
        max_retries (int): The maximum number of retries to use for the code generation.
        temperature (float): The temperature to use for the code generation.
        tools (Optional[List[Union[Callable, dict, BaseModel]]]): The tools to use for the code generation.
        verbose (bool): Whether to print the code generation process.
        mode (Literal["json", "md_json", "tools"]): The mode to use for the code generation.

    Returns:
        The generated code.
    """

    from ..main import Client

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
        mode="md_json" if model.startswith(("ollama/", "ollama_chat/")) else mode,
    )
    return response.code
