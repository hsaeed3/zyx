from pydantic import BaseModel
from typing import Optional, Literal


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
    mode: Literal["json", "md_json", "tools"] = "md_json",
    verbose: bool = False,
):
    from ..completion import CompletionClient

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

    response = CompletionClient().completion(
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
        response_model=target,
    )

    return response
