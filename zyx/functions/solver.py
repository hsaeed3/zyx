from pydantic import BaseModel
from typing import List, Literal, Optional, Union
from rich.progress import Progress, SpinnerColumn, TextColumn


from ..resources.types import completion_create_params as params
from ..base_client import Client
from .._rich import logger


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
    high_level_concept: bool = False,
    tree_of_thought: bool = False,
    max_depth: int = 3,
    branching_factor: int = 3,
    model: Union[str, params.ChatModel] = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0.7,
    mode: params.InstructorMode = "tool_call",
    process: Literal["single", "batch"] = "single",
    batch_size: int = 3,
    progress_bar: Optional[bool] = True,
    verbose: bool = False,
    provider: Optional[Literal["litellm", "openai"]] = "openai",
    client: Optional[Client] = None,
) -> Union[FinalAnswer, TreeOfThoughtResult]:
    """
    Solves a problem using a combination of chain-of-thought, high-level concept, and tree-of-thought approaches.
    """

    if verbose:
        logger.info(f"Solving problem: {problem}")
        logger.info(f"Using model: {model}")
        logger.info(f"Process: {process}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Use high-level concept: {high_level_concept}")
        logger.info(f"Use tree of thought: {tree_of_thought}")

    completion_client = Client(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        provider=provider,
        verbose=verbose,
    ) if client is None else client

    if progress_bar:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task_id = progress.add_task("Initializing Solver...", total=None)

            if high_level_concept:
                progress.update(task_id, description="Generating High-Level Concept...", completed=0)
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
                    progress_bar=False,
                )

                high_level_concept = high_level_concept_response.concept
                progress.update(task_id, description="High-Level Concept Generated", completed=1)

                progress.update(task_id, description="Generating Final Answer...", completed=0)
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
                    progress_bar=False,
                )

                progress.update(task_id, description="Final Answer Generated", completed=1)
                return final_answer_response

            if tree_of_thought:
                progress.update(task_id, description="Generating Tree of Thought...", completed=0)

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
                        progress_bar=False,
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
                progress.update(task_id, description="Tree of Thought Generated", completed=1)

                progress.update(task_id, description="Generating Final Answer from Tree...", completed=0)
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
                    progress_bar=False,
                )

                progress.update(task_id, description="Final Answer from Tree Generated", completed=1)
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
        progress_bar=False,
    )

    final_answer = response.thoughts[
        -1
    ].content  # Assuming the last thought is the final answer

    return FinalAnswer(answer=final_answer)


if __name__ == "__main__":
    result = solve(
        "What is the significance of the Pythagorean theorem in mathematics?",
        verbose=True,
        high_level_concept=True,
        use_tree_of_thought=True,
        batch_size=1,
    )
    print(f"Final answer: {result.answer}")
    if isinstance(result, TreeOfThoughtResult):
        print(f"Reasoning tree: {result.reasoning_tree.model_dump_json(indent=2)}")