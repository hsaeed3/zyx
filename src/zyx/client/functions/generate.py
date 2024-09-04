__all__ = ["generate"]

from ...core.main import BaseModel
from typing import Optional, Literal, List


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
    mode: Literal["json", "md_json", "tools"] = "md_json",
    verbose: bool = False,
) -> List:
    """
    Generates a list of instances of the given Pydantic model.

    Example:
        ```python
        import zyx

        class User(BaseModel):
            name: str
            age: int

        zyx.generate(User, n=5)
        ```

    Parameters:
        target (BaseModel): The Pydantic model to generate instances of.
        instructions (Optional[str]): The instructions for the generator.
        n (int): The number of instances to generate.
        model (str): The model to use for generation.
        api_key (Optional[str]): The API key to use for generation.
        base_url (Optional[str]): The base URL to use for generation.
        organization (Optional[str]): The organization to use for generation.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries to make.
        temperature (float): The temperature to use for generation.
        mode (Literal["json", "md_json", "tools"]): The mode to use for generation.
        verbose (bool): Whether to print verbose output.

    Returns:
        List[BaseModel]: A list of instances of the given Pydantic model.
    """

    from ..main import Client
    from pydantic import create_model

    ResponseModel = create_model("ResponseModel", items=(List[target], ...))

    system_message = f"""
    You are a data generator. Your task is to generate {n} valid instance(s) of the following Pydantic model:
    
    {target.model_json_schema()}
    
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
        mode="md_json" if model.startswith(("ollama/", "ollama_chat/")) else mode,
        response_model=ResponseModel,
    )
    return response.items
