from pydantic import BaseModel
from typing import Optional, Literal, List


class Thought(BaseModel):
    content: str
    score: float


class Thoughts(BaseModel):
    thoughts: List[Thought]


class TreeNode(BaseModel):
    thought: Thought
    children: List["TreeNode"] = []


class TreeOfThoughtResult(BaseModel):
    final_answer: str
    reasoning_tree: TreeNode


def tree_of_thought(
    problem: str,
    max_depth: int = 3,
    branching_factor: int = 3,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0.7,
    mode: Literal["json", "md_json", "tools"] = "md_json",
    verbose: bool = False,
) -> TreeOfThoughtResult:
    """
    Implements the Tree-of-Thought prompting technique to solve complex problems.

    Example:
        ```python
        result = zyx.tree_of_thought("Solve the equation: 2x + 5 = 13")
        print(result.final_answer)
        print(result.reasoning_tree)
        ```

    Parameters:
        problem (str): The problem to solve.
        max_depth (int): The maximum depth of the reasoning tree.
        branching_factor (int): The number of thoughts to generate at each step.
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
        TreeOfThoughtResult: The final answer and the reasoning tree.
    """
    from ..completion import CompletionClient, ClientParams

    client = CompletionClient()

    def generate_thoughts(current_problem: str, depth: int) -> TreeNode:
        if depth >= max_depth:
            return TreeNode(thought=Thought(content="Reached max depth", score=0))

        system_message = f"""
        ## CONTEXT ## \n
        You are a problem-solving assistant using the Tree-of-Thought method. Tree of Thoughts (ToT), is a paradigm that allows LMs to explore multiple reasoning paths over thoughts. \n\n
        
        ## INSTRUCTIONS ## \n
        - Generate {branching_factor} possible next steps or thoughts for solving the given problem.
        - For each thought, also provide a score between 0 and 1 indicating how promising it is for solving the problem.
        """

        user_message = f"Problem: {current_problem}\nDepth: {depth}\nGenerate {branching_factor} thoughts:"

        params = ClientParams(
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
            mode=mode,
            response_model=Thoughts,
            kwargs={},
        )

        response = client.completion(params=params, verbose=verbose)

        best_thought = max(response.thoughts, key=lambda x: x.score)
        node = TreeNode(thought=best_thought)

        if depth < max_depth - 1:
            for _ in range(branching_factor):
                child = generate_thoughts(
                    f"{current_problem}\nPrevious thought: {best_thought.content}",
                    depth + 1,
                )
                node.children.append(child)

        return node

    root = generate_thoughts(problem, 0)

    system_message = f"""
    ## CONTEXT ## \n
    You are a problem-solving assistant using the Tree-of-Thought method. Tree of Thoughts (ToT), is a paradigm that allows LMs to explore multiple reasoning paths over thoughts. \n\n
    
    ## INSTRUCTIONS ## \n
    - Generate {branching_factor} possible next steps or thoughts for solving the given problem.
    - For each thought, also provide a score between 0 and 1 indicating how promising it is for solving the problem.
    """

    user_message = f"Problem: {problem}\nReasoning tree: {root.model_dump_json()}\nProvide the final answer:"

    class FinalAnswer(BaseModel):
        answer: str

    params = ClientParams(
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
        mode=mode,
        response_model=FinalAnswer,
        kwargs={},  # Add an empty dict for kwargs
    )

    final_answer = client.completion(params=params, verbose=verbose)

    return TreeOfThoughtResult(final_answer=final_answer.answer, reasoning_tree=root)


if __name__ == "__main__":
    result = tree_of_thought(
        "Solve the equation: 2x + 5 = 13", verbose=True, model="gpt-4o-mini"
    )
    print(f"Final answer: {result.final_answer}")
    print(f"Reasoning tree: {result.reasoning_tree.model_dump_json(indent=2)}")
