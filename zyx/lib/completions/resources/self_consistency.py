from pydantic import BaseModel
from typing import Optional, Literal, List
from collections import Counter
from ...types import client as clienttypes


class ReasoningPath(BaseModel):
    steps: List[str]
    final_answer: str


class SelfConsistencyResult(BaseModel):
    final_answer: str
    confidence: float
    reasoning_paths: List[ReasoningPath]


def self_consistency(
    problem: str,
    num_paths: int = 5,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0.7,
    mode: clienttypes.InstructorMode = "markdown_json_mode",
    verbose: bool = False,
) -> SelfConsistencyResult:
    """
    Implements the Self-Consistency prompting technique to solve problems with higher confidence.

    Example:
        ```python
        result = zyx.self_consistency("What is the capital of France?")
        print(result.final_answer)
        print(result.confidence)
        ```

    Parameters:
        problem (str): The problem to solve.
        num_paths (int): The number of reasoning paths to generate.
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
        SelfConsistencyResult: The final answer, confidence, and reasoning paths.
    """
    from ..client import completion

    system_message = """
    ## CONTEXT ## \n
    You are a problem-solving assistant using the Chain of Thought Self Consistency method.
    Self-Consistency is a technique that involves generating multiple reasoning paths and selecting the most consistent answer.
    This method helps to increase the confidence in the final answer by reducing the risk of hallucinations. \n\n
    
    ## INSTRUCTIONS ## \n
    - Generate {num_paths} reasoning paths for the given problem.
    - For each reasoning path, provide a step-by-step reasoning path to solve the problem, then give your final answer.
    - Do not hallucinate or make up a reasoning path.
    """

    user_message = f"Problem: {problem}\nProvide your reasoning steps and final answer:"

    reasoning_paths = []

    for _ in range(num_paths):
        params = clienttypes.CompletionArgs(
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
            response_model=ReasoningPath,
            kwargs={},
        )

        response = completion(params=params, mode=mode, verbose=verbose)
        reasoning_paths.append(response)

    # Count the occurrences of each final answer
    answer_counts = Counter(path.final_answer for path in reasoning_paths)

    # Find the most common answer and its count
    final_answer, max_count = answer_counts.most_common(1)[0]

    # Calculate confidence
    confidence = max_count / num_paths

    return SelfConsistencyResult(
        final_answer=final_answer,
        confidence=confidence,
        reasoning_paths=reasoning_paths,
    )


if __name__ == "__main__":
    result = self_consistency("What is the capital of France?", verbose=True)
    print(f"Final answer: {result.final_answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Number of reasoning paths: {len(result.reasoning_paths)}")