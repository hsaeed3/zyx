from pydantic import BaseModel
from typing import Optional, Literal
from ...types import client as clienttypes


class StepBackResult(BaseModel):
    high_level_concept: str
    final_answer: str


class HighLevelConcept(BaseModel):
    concept: str


class FinalAnswer(BaseModel):
    answer: str


def step_back(
    problem: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0.2,
    mode: clienttypes.InstructorMode = "markdown_json_mode",
    verbose: bool = False,
) -> StepBackResult:
    """
    Implements the Step-Back prompting technique to solve problems by first considering high-level concepts.

    Example:
        ```python
        result = zyx.step_back("What is the significance of the Pythagorean theorem in mathematics?")
        print(f"High-level concept: {result.high_level_concept}")
        print(f"Final answer: {result.final_answer}")
        ```

    Parameters:
        problem (str): The problem to solve.
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
        StepBackResult: The high-level concept and the final answer.
    """
    from .... import completion

    # Step 1: Generate high-level concept
    system_message = """
## CONTEXT ## \n
You are a problem-solving assistant using the Step-Back method.
Given a specific problem, provide a high-level concept or principle that relates to this problem.
This concept should be more general and abstract than the original problem. \n\n

## INSTRUCTIONS ## \n
- Reason about the problem and identify the core concept or principle that is relevant to solving it.
- Generate a high-level concept that is more general and abstract than the original problem.
- Do not hallucinate or make up a concept.
    """

    user_message = (
        f"Problem: {problem}\nProvide a high-level concept related to this problem:"
    )

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
        response_model=HighLevelConcept,
        kwargs={},
    )

    high_level_concept_response = completion(params=params, mode=mode, verbose=verbose)
    high_level_concept = high_level_concept_response.concept

    # Step 2: Generate final answer
    system_message = """
## CONTEXT ## \n
You are a problem-solving assistant using the Step-Back method.
Given a specific problem, provide a high-level concept or principle that relates to this problem.
This concept should be more general and abstract than the original problem. \n\n

## INSTRUCTIONS ## \n
- Reason about the problem and identify the core concept or principle that is relevant to solving it.
- Generate a high-level concept that is more general and abstract than the original problem.
- Do not hallucinate or make up a concept.
    """

    user_message = f"Problem: {problem}\nHigh-level concept: {high_level_concept}\nProvide a comprehensive answer to the original problem:"

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
        response_model=FinalAnswer,
        kwargs={},
    )

    final_answer_response = completion(params=params, mode=mode, verbose=verbose)
    final_answer = final_answer_response.answer

    return StepBackResult(
        high_level_concept=high_level_concept, final_answer=final_answer
    )


if __name__ == "__main__":
    result = step_back(
        "What is the significance of the Pythagorean theorem in mathematics?",
        verbose=True,
    )
    print(f"High-level concept: {result.high_level_concept}")
    print(f"Final answer: {result.final_answer}")