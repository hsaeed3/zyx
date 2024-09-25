from ...types import client as clienttypes
from pydantic import BaseModel
from typing import Optional, Literal


def extract(
    target: BaseModel,
    text: str,
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
):
    
    """An LLM abstraction for extracting structured information from text.
    
    Example:
        ```python
        import zyx

        class User(BaseModel):
            name: str
            age: int
        zyx.llm.extract(User, "John is 20 years old")
        ```
    """

    from .... import completion

    system_message = f"""
    You are an information extractor. Your task is to extract relevant information from the given text 
    and fit it into the following Pydantic model:
    
    {target.model_json_schema()}
    
    Instructions:
    - Only Extract information from the text and fit it into the given model.
    - Do not infer or generate any information that is not present in the input text.
    - If a required field cannot be filled with information from the text, leave it as None or an empty string as appropriate.
    """

    user_message = f"Extract information from the following text and fit it into the given model:\n\n{text}"

    response = completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        client=client,
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

    return response