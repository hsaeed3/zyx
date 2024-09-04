__all__ = ["chainofthought"]

from ...core.main import BaseModel
from typing import Any, Optional, Literal


def chainofthought(
    query: str,
    answer_type: Any = str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0,
    verbose: bool = False,
    mode: Literal["json", "md_json", "tools"] = "md_json",
):
    """Module for invoking quick chain of thought prompting.

    Example:
        ```python
        import zyx

        zyx.chainofthought(query="What is the capital of the moon?")
        ```

    Parameters:
        query (str): The query to be processed by the chain of thought.
        answer_type (Any): The type of the answer to be returned.
        model (str): The model to be used for the chain of thought.
        api_key (Optional[str]): The API key to be used for the chain of thought.
        base_url (Optional[str]): The base URL to be used for the chain of thought.
        max_tokens (Optional[int]): The maximum number of tokens to be used for the chain of thought.
        temperature (float): The temperature to be used for the chain of thought.
        verbose (bool): Whether to print the chain of thought.
        mode (Literal["json", "md_json", "tools"]): The mode to be used for the chain of thought.
    """

    from ..main import Client

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

    reasoning_response = Client().completion(
        messages=[{"role": "user", "content": reasoning_prompt}],
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        response_model=Reasoning,
        verbose=verbose,
        mode="md_json" if model.startswith(("ollama/", "ollama_chat/")) else mode,
    )

    if verbose:
        print("Chain of Thought:")
        print(reasoning_response.chain_of_thought)

    extraction_prompt = f"""
    Based on the following reasoning:
    {reasoning_response.chain_of_thought}

    Provide the final answer to the question: "{query}"
    Your answer should be of type: {answer_type.__name__}
    Only provide the final answer, without any additional explanation.
    """

    final_answer_response = Client().completion(
        messages=[{"role": "user", "content": extraction_prompt}],
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        response_model=FinalAnswer,
        mode="md_json" if model.startswith(("ollama/", "ollama_chat/")) else mode,
        verbose=verbose,
    )

    return final_answer_response.answer
