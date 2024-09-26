from pydantic import BaseModel
from typing import Optional, List
from ...types import client as clienttypes


class SubProblem(BaseModel):
    description: str
    solution: Optional[str] = None


class LeastToMostResult(BaseModel):
    final_answer: str
    sub_problems: List[SubProblem]


def least_to_most(
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
) -> LeastToMostResult:
    """
    Implements the Least-to-Most prompting technique to break down and solve complex problems.

    Example:
        ```python
        result = zyx.least_to_most("Explain the process of photosynthesis in plants.")
        print(result.final_answer)
        for sub_problem in result.sub_problems:
            print(f"Sub-problem: {sub_problem.description}")
            print(f"Solution: {sub_problem.solution}")
        ```

    Parameters:
        problem (str): The complex problem to solve.
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
        LeastToMostResult: The final answer and the list of sub-problems with their solutions.
    """
    from ..client import completion

    # Step 1: Break down the problem into sub-problems
    system_message = """
    You are a problem-solving assistant using the Least-to-Most method.
    Break down the given complex problem into smaller, manageable sub-problems.
    List these sub-problems in order, from the simplest to the most complex.
    """

    user_message = f"Problem: {problem}\nBreak this down into sub-problems:"

    sub_problems_response = completion(
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
        response_model=List[str],
    )

    sub_problems = [SubProblem(description=desc) for desc in sub_problems_response]

    # Step 2: Solve each sub-problem sequentially
    for i, sub_problem in enumerate(sub_problems):
        system_message = """
        You are a problem-solving assistant. Solve the given sub-problem based on the context provided.
        """

        context = "\n".join(
            [
                f"{j+1}. {sp.description}: {sp.solution}"
                for j, sp in enumerate(sub_problems[:i])
            ]
        )
        user_message = (
            f"Context:\n{context}\n\nSub-problem to solve: {sub_problem.description}"
        )

        solution = completion(
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
            response_model=str,
        )

        sub_problems[i].solution = solution

    # Step 3: Generate final answer
    system_message = """
    You are a problem-solving assistant. Given the original problem and the solutions to its sub-problems,
    provide a comprehensive final answer.
    """

    sub_problems_summary = "\n".join(
        [f"{i+1}. {sp.description}: {sp.solution}" for i, sp in enumerate(sub_problems)]
    )
    user_message = f"Original problem: {problem}\n\nSub-problems and solutions:\n{sub_problems_summary}\n\nProvide a comprehensive final answer:"

    final_answer = completion(
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
        response_model=str,
    )

    return LeastToMostResult(final_answer=final_answer, sub_problems=sub_problems)


if __name__ == "__main__":
    result = least_to_most(
        "Explain the process of photosynthesis in plants.", verbose=True
    )
    print(f"Final answer: {result.final_answer}")
    for i, sub_problem in enumerate(result.sub_problems):
        print(f"Sub-problem {i+1}: {sub_problem.description}")
        print(f"Solution: {sub_problem.solution}\n")