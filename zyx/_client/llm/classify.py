from pydantic import BaseModel
from typing import Optional, List, Union, Literal


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
    mode: Literal["json", "md_json", "tools"] = "md_json",
    verbose: bool = False,
) -> List:
    """
    Classifies the given text(s) into one of the given labels.

    Example:
        ```python
        zyx.classify("I love programming.", ["positive", "negative"])
        ```

    Parameters:
        inputs (Union[str, List[str]]): The text(s) to classify.
        labels (List[str]): The labels to classify the text(s) into.
        n (int): The number of classifications to provide for each input.
        model (str): The model to use for classification.
        api_key (Optional[str]): The API key to use for classification.
        base_url (Optional[str]): The base URL to use for classification.
        organization (Optional[str]): The organization to use for classification.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries to make.
        temperature (float): The temperature to use for classification.
        mode (Literal["json", "md_json", "tools"]): The mode to use for classification.
        verbose (bool): Whether to print verbose output.

    Returns:
        List[List[ClassificationResult]]: The classifications for each input.
    """
    from ..completion import CompletionClient
    from pydantic import create_model

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
