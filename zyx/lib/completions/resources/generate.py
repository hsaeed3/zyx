from ...types import client as clienttypes
from pydantic import BaseModel
from typing import Optional, Literal, List


def generate(
    target: BaseModel,
    instructions: Optional[str] = None,
    n: int = 1,
    process: Literal["multi", "sequential"] = "multi", 
    client: Literal["litellm", "openai"] = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0,
    mode: clienttypes.InstructorMode = "markdown_json_mode",
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

        zyx.llm.generate(User, n=5)
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
        process (Literal["multi", "sequential"]): The generation process to use when n > 1.
            "multi" generates all instances at once, "sequential" generates them one by one.

    Returns:
        List[BaseModel]: A list of instances of the given Pydantic model.
    """

    from ..client import completion
    from pydantic import create_model

    if n == 1:
        ResponseModel = target
    else:
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

    # Function to generate a single instance
    def generate_single(previous_instances=None):
        user_message = instructions if instructions else f"Generate 1 instance of the given model."
        if previous_instances:
            user_message += f"\nPreviously generated instances: {previous_instances}\nEnsure this new instance is different."

        return completion(
            client=client,
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
            mode="markdown_json_mode" if model.startswith(("ollama/", "ollama_chat/")) else mode,
            response_model=target,
        )

    if n == 1 or process == "multi":
        # Existing logic for single or multi-generation
        user_message = instructions if instructions else f"Generate {n} instance(s) of the given model."
        response = completion(
            client=client,
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
            mode="markdown_json_mode" if model.startswith(("ollama/", "ollama_chat/")) else mode,
            response_model=ResponseModel,
        )
        return response if n == 1 else response.items
    else:  # Sequential generation
        results = []
        for _ in range(n):
            result = generate_single(previous_instances=results[-3:] if results else None)
            results.append(result)
        return results


if __name__ == "__main__":
    import zyx

    class User(BaseModel):
        name: str
        age: int

    print(zyx.generate(User))