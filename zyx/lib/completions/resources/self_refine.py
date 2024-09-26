from pydantic import BaseModel
from typing import Optional, List, Literal
from ...types import client as clienttypes


class RefinementStep(BaseModel):
    answer: str
    feedback: str


class SelfRefineResult(BaseModel):
    final_answer: str
    refinement_steps: List[RefinementStep]


class Answer(BaseModel):
    content: str


class Feedback(BaseModel):
    content: str


def self_refine(
    problem: str,
    max_iterations: int = 3,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0.2,
    mode: clienttypes.InstructorMode = "markdown_json_mode",
    verbose: bool = False,
) -> SelfRefineResult:
    """
    Implements the Self-Refine technique to iteratively improve answers to a given problem.

    Example:
        ```python
        result = zyx.self_refine("Explain the concept of quantum entanglement.")
        print(f"Final answer: {result.final_answer}")
        for i, step in enumerate(result.refinement_steps):
            print(f"Step {i+1}:")
            print(f"Answer: {step.answer}")
            print(f"Feedback: {step.feedback}\n")
        ```

    Parameters:
        problem (str): The problem to solve.
        max_iterations (int): The maximum number of refinement iterations.
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
        SelfRefineResult: The final answer and the list of refinement steps.
    """
    from ..client import completion

    refinement_steps = []

    # Initial answer
    system_message = (
        "You are a knowledgeable assistant. Provide an answer to the given problem."
    )
    user_message = f"Problem: {problem}\nProvide an answer:"

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
        response_model=Answer,
        kwargs={},
    )

    current_answer = completion(params=params, mode=mode, verbose=verbose).content

    for _ in range(max_iterations):
        # Generate feedback
        system_message = """
        You are a critical reviewer. Analyze the given answer and provide constructive feedback.
        Focus on areas that need improvement, clarification, or expansion.
        """
        user_message = (
            f"Problem: {problem}\nCurrent answer: {current_answer}\nProvide feedback:"
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
            response_model=Feedback,
            kwargs={},
        )

        feedback = completion(params=params, mode=mode, verbose=verbose).content

        refinement_steps.append(
            RefinementStep(answer=current_answer, feedback=feedback)
        )

        # Generate improved answer
        system_message = """
        You are a knowledgeable assistant. Improve the given answer based on the feedback provided.
        """
        user_message = f"Problem: {problem}\nCurrent answer: {current_answer}\nFeedback: {feedback}\nProvide an improved answer:"

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
            response_model=Answer,
            kwargs={},
        )

        current_answer = completion(params=params, mode=mode, verbose=verbose).content

    return SelfRefineResult(
        final_answer=current_answer, refinement_steps=refinement_steps
    )


if __name__ == "__main__":
    result = self_refine("Explain the concept of quantum entanglement.", verbose=True)
    print(f"Final answer: {result.final_answer}")
    for i, step in enumerate(result.refinement_steps):
        print(f"Step {i+1}:")
        print(f"Answer: {step.answer}")
        print(f"Feedback: {step.feedback}\n")
