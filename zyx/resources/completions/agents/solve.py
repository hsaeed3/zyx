from pydantic import BaseModel
from typing import List, Literal, Optional, Union
from ....lib.utils.logger import get_logger
from ....client import Client, InstructorMode

logger = get_logger("solve")


class Thought(BaseModel):
    content: str
    score: float


class Thoughts(BaseModel):
    thoughts: List[Thought]


class HighLevelConcept(BaseModel):
    concept: str


class FinalAnswer(BaseModel):
    answer: str


class TreeNode(BaseModel):
    thought: Thought
    children: List["TreeNode"] = []


class TreeOfThoughtResult(BaseModel):
    final_answer: str
    reasoning_tree: TreeNode


def solve(
    problem: str,
    use_high_level_concept: bool = False,
    use_tree_of_thought: bool = False,
    max_depth: int = 3,
    branching_factor: int = 3,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0.7,
    mode: InstructorMode = "markdown_json_mode",
    process: Literal["single", "batch"] = "single",
    batch_size: int = 3,
    verbose: bool = False,
    client: Literal["litellm", "openai"] = "openai",
) -> Union[FinalAnswer, TreeOfThoughtResult]:
    """
    Solves a problem using a combination of chain-of-thought, high-level concept, and tree-of-thought approaches.

    Parameters:
        problem (str): The problem to solve.
        model (str): The model to use for generation.
        api_key (Optional[str]): The API key to use for generation.
        base_url (Optional[str]): The base URL to use for generation.
        organization (Optional[str]): The organization to use for generation.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries to make.
        temperature (float): The temperature to use for generation.
        mode (InstructorMode): The mode to use for generation.
        process (Literal["single", "batch"]): The process type (single or batch).
        batch_size (int): The batch size for processing.
        verbose (bool): Whether to print verbose output.
        use_high_level_concept (bool): Whether to use the high-level concept approach.
        use_tree_of_thought (bool): Whether to use the tree-of-thought approach.
        max_depth (int): The maximum depth of the reasoning tree (if using tree-of-thought).
        branching_factor (int): The number of thoughts to generate at each step (if using tree-of-thought).

    Returns:
        Union[FinalAnswer, TreeOfThoughtResult]: The final answer or the tree of thought result.
    """

    if verbose:
        logger.info(f"Solving problem: {problem}")
        logger.info(f"Using model: {model}")
        logger.info(f"Process: {process}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Use high-level concept: {use_high_level_concept}")
        logger.info(f"Use tree of thought: {use_tree_of_thought}")

    completion_client = Client(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        provider=client,
        verbose=verbose,
    )

    if use_high_level_concept:
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

        high_level_concept_response = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=model,
            response_model=HighLevelConcept,
            mode=mode,
            max_retries=max_retries,
            temperature=temperature,
        )

        high_level_concept = high_level_concept_response.concept

        # Step 2: Generate final answer based on high-level concept
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

        final_answer_response = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=model,
            response_model=FinalAnswer,
            mode=mode,
            max_retries=max_retries,
            temperature=temperature,
        )

        return final_answer_response

    if use_tree_of_thought:

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

            response = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                response_model=Thoughts,
                mode=mode,
                max_retries=max_retries,
                temperature=temperature,
            )

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

        final_answer_response = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=model,
            response_model=FinalAnswer,
            mode=mode,
            max_retries=max_retries,
            temperature=temperature,
        )

        return TreeOfThoughtResult(
            final_answer=final_answer_response.answer, reasoning_tree=root
        )

    # Default to chain-of-thought approach
    system_message = f"""
    ## CONTEXT ## \n
    You are a problem-solving assistant. Use a chain-of-thought approach to solve the given problem step by step. \n\n
    
    ## INSTRUCTIONS ## \n
    - Generate a sequence of thoughts to solve the problem.
    - Each thought should build on the previous one.
    - Do not hallucinate or make up information.
    """

    user_message = (
        f"Problem: {problem}\nGenerate a sequence of thoughts to solve the problem:"
    )

    response = completion_client.completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        model=model,
        response_model=Thoughts,
        mode=mode,
        max_retries=max_retries,
        temperature=temperature,
    )

    final_answer = response.thoughts[
        -1
    ].content  # Assuming the last thought is the final answer

    return FinalAnswer(answer=final_answer)


if __name__ == "__main__":
    result = solve(
        "What is the significance of the Pythagorean theorem in mathematics?",
        verbose=True,
        use_high_level_concept=True,
        use_tree_of_thought=True,
        batch_size=1,
    )
    print(f"Final answer: {result.answer}")
    if isinstance(result, TreeOfThoughtResult):
        print(f"Reasoning tree: {result.reasoning_tree.model_dump_json(indent=2)}")
