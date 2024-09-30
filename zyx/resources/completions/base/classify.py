from ....lib.utils.logger import get_logger
from ....client import Client, InstructorMode
from pydantic import BaseModel, create_model
from typing import List, Literal, Optional, Union


logger = get_logger("classify")


def classify(
    inputs: Union[str, List[str]],
    labels: List[str],
    classification: Literal["single", "multi"] = "single",
    n: int = 1,
    batch_size: int = 3,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    mode: InstructorMode = "tool_call",
    temperature: Optional[float] = None,
    client: Optional[Literal["openai", "litellm"]] = None,
    verbose: bool = False,
) -> List:
    """
    Classifies given input(s) into one or more of the provided labels.

    Examples:
        >>> classify(["I love programming in Python", "I like french fries", "I love programming in Julia"], ["code", "food"], classification = "single", batch_size = 2, verbose = True)
        [
            ClassificationResult(text="I love programming in Python", label="code"),
            ClassificationResult(text="I like french fries", label="food")
        ]

    Args:
        inputs (Union[str, List[str]]): The input text to classify.
        labels (List[str]): The list of labels to classify the input text into.
        classification (Literal["single", "multi"]): The type of classification to perform. Defaults to "single".
        n (int): The number of classifications to generate. Defaults to 1.
        batch_size (int): The number of inputs to classify at a time. Defaults to 3.
        model (str): The model to use for classification. Defaults to "gpt-4o-mini".
        api_key (Optional[str]): The API key to use for OpenAI. Defaults to None.
        base_url (Optional[str]): The base URL for the OpenAI API. Defaults to None.
        organization (Optional[str]): The organization to use for OpenAI. Defaults to None.
        mode (InstructorMode): The mode to use for classification. Defaults to "tool_call".
        temperature (Optional[float]): The temperature to use for classification. Defaults to None.
        client (Optional[Literal["openai", "litellm"]]): The client to use for classification. Defaults to None.
        verbose (bool): Whether to print verbose output. Defaults to False.

    Returns:
        List[Union[ClassificationResult, MultiClassificationResult]]: The classifications generated.
    """

    if verbose:
        logger.info(f"Classifying {len(inputs)} inputs into {len(labels)} labels.")
        logger.info(f"Using model: {model}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Classification Mode: {classification}")

    class ClassificationResult(BaseModel):
        text: str
        label: str

    class MultiClassificationResult(BaseModel):
        text: str
        labels: List[str]

    if classification == "single":
        system_message = f"""
You are a world class text classifier. Your only task is to classify given text into the following categories:
{', '.join(labels)}

For each input, you must only provide one label for classification. Each classification you generate must contain
both the text and the label you have classified it under.
"""
        if batch_size == 1:
            response_model = ClassificationResult
        elif batch_size > 1:
            response_model = create_model(
                "ClassificationResult", items=(List[ClassificationResult], ...)
            )
        else:
            raise ValueError("Batch size must be a positive integer.")
    else:
        system_message = f"""
You are a world class text classifier. Your only task is to classify given text into the following categories:
{', '.join(labels)}

For each input, you must provide all labels that apply. Each classification you generate must contain
both the text and the label you have classified it under.
"""
        if batch_size == 1:
            response_model = MultiClassificationResult
        elif batch_size > 1:
            response_model = create_model(
                "ClassificationResult", items=(List[MultiClassificationResult], ...)
            )
        else:
            raise ValueError("Batch size must be a positive integer.")

    if isinstance(inputs, str):
        inputs = [inputs]

    results = []

    completion_client = Client(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        provider=client,
        verbose=verbose,
    )

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]

        user_message = "Classify the following text(s):\n\n"
        for idx, text in enumerate(batch, 1):
            user_message += f"{idx}. {text}\n\n"

        result = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=model,
            response_model=response_model,
            mode=mode,
            temperature=temperature,
        )

        if batch_size == 1:
            results.append(result)
        else:
            results.extend(result.items)

    return results if len(results) > 1 else results[0]


if __name__ == "__main__":
    items = [
        "I love programming in Python",
        "I like french fries",
        "I love programming in Julia",
    ]

    labels = ["code", "food"]

    print(classify(items, labels, classification="single", batch_size=2, verbose=True))
