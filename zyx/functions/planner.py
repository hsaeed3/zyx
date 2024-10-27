from pydantic import BaseModel, create_model, Field
from typing import Optional, List, Union, Literal, Type, Any
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..resources.types import completion_create_params as params
from ..base_client import Client
from .._rich import logger


class Task(BaseModel):
    description: str
    details: Optional[str] = None


class Plan(BaseModel):
    tasks: List[Task]


def planner(
    input: Union[str, Type[BaseModel]],
    instructions: Optional[str] = None,
    process: Literal["single", "batch"] = "single",
    n: int = 1,
    batch_size: int = 3,
    steps: int = 5,
    model: Union[str, params.ChatModel] = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    temperature: Optional[float] = None,
    mode: params.InstructorMode = "tool_call",
    max_retries: int = 3,
    provider: Optional[Literal["openai", "litellm"]] = "openai",
    client: Optional[Client] = None,
    verbose: bool = False,
    progress_bar: Optional[bool] = True,
) -> Union[Plan, List[Plan], Any, List[Any]]:
    """
    Generates a plan or batch of plans based on the input using the Tree of Thoughts method.

    Example:
        ```python
        >>> plan(
            input="Create a marketing strategy for a new smartphone",
            steps=5,
            verbose=True
        )
        ```

    Args:
        input (Union[str, Type[BaseModel]]): The input can be either a string describing the task or a Pydantic model class.
        instructions (Optional[str]): Additional instructions for the planning process.
        process (Literal["single", "batch"]): Process can be either "single" or "batch".
        n (int): Number of plans to generate.
        batch_size (int): Number of plans to generate in a single batch.
        steps (int): Number of steps per plan.
        model (str): The model to use for the planning process.
        api_key (Optional[str]): The API key to use for the planning process.
        base_url (Optional[str]): The base URL to use for the planning process.
        organization (Optional[str]): The organization to use for the planning process.
        temperature (Optional[float]): The temperature to use for the planning process.
        mode (InstructorMode): The mode to use for the planning process.
        max_retries (int): The maximum number of retries to use for the planning process.
        client (Optional[Literal["openai", "litellm"]]): The client to use for the planning process.
        verbose (bool): Whether to print the planning process to the console.

    Returns:
        Union[Plan, List[Plan], Any, List[Any]]: The plan or batch of plans.
    """

    if verbose:
        logger.info(f"Generating {n} plan(s) using Tree of Thoughts method")
        logger.info(f"Using model: {model}")
        logger.info(f"Number of steps per plan: {steps}")
        logger.info(f"Process: {process}")
        logger.info(f"Batch size: {batch_size}")

    completion_client = Client(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        provider=provider,
        verbose=verbose,
    ) if client is None else client

    if isinstance(input, str):
        system_message = _get_string_system_message(input, steps)
        response_model = Plan
    elif isinstance(input, type) and issubclass(input, BaseModel):
        system_message = _get_model_system_message(input, steps)
        response_model = create_model(f"{input.__name__}Plan", tasks=(List[input], ...))
    else:
        raise ValueError("Input must be either a string or a Pydantic model class.")

    user_message = (
        instructions if instructions else f"Generate a plan with {steps} steps."
    )

    if progress_bar:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task_id = progress.add_task("Planning...", total=n)

            if process == "single" or n == 1:
                result = completion_client.completion(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    model=model,
                    response_model=response_model,
                    mode=mode,
                    max_retries=max_retries,
                    temperature=temperature,
                    progress_bar=False,
                )
                progress.update(task_id, completed=n)
                return result
            else:  # batch process
                batch_response_model = create_model(
                    "ResponseModel", items=(List[response_model], ...)
                )
                results = []
                for i in range(0, n, batch_size):
                    batch_n = min(batch_size, n - i)
                    batch_message = f"Generate {batch_n} plans, each with {steps} steps."
                    if results:
                        batch_message += f"\nPreviously generated plans: {results[-3:]}\nEnsure these new plans are different."

                    result = completion_client.completion(
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": batch_message},
                        ],
                        model=model,
                        response_model=batch_response_model,
                        mode=mode,
                        max_retries=max_retries,
                        temperature=temperature,
                        progress_bar=False,
                    )

                    results.extend(result.items)
                    progress.update(task_id, advance=batch_n)

                return results
    else:
        if process == "single" or n == 1:
            result = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                response_model=response_model,
                mode=mode,
                max_retries=max_retries,
                temperature=temperature,
                progress_bar=False,
            )
            return result
        else:  # batch process
            batch_response_model = create_model(
                "ResponseModel", items=(List[response_model], ...)
            )
            results = []
            for i in range(0, n, batch_size):
                batch_n = min(batch_size, n - i)
                batch_message = f"Generate {batch_n} plans, each with {steps} steps."
                if results:
                    batch_message += f"\nPreviously generated plans: {results[-3:]}\nEnsure these new plans are different."

                result = completion_client.completion(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": batch_message},
                    ],
                    model=model,
                    response_model=batch_response_model,
                    mode=mode,
                    max_retries=max_retries,
                    temperature=temperature,
                    progress_bar=False,
                )

                results.extend(result.items)

            return results


def _get_string_system_message(input: str, steps: int) -> str:
    return f"""
    You are a planning assistant using the Tree of Thoughts method. Your task is to generate a detailed plan based on the given input.
    Follow these steps:
    1. Generate 3 high-level approaches to achieve the goal: {input}
    2. For each approach, create 3 potential outcomes (best-case, average-case, worst-case).
    3. Evaluate the outcomes and select the most promising approach.
    4. Break down the selected approach into {steps} detailed, actionable tasks.
    Return the tasks as a Plan object with a list of Task objects.
    """


def _get_model_system_message(input_model: Type[BaseModel], steps: int) -> str:
    return f"""
    You are a planning assistant using the Tree of Thoughts method. Your task is to generate a detailed plan based on the given Pydantic model.
    Follow these steps:
    1. Analyze the provided model structure.
    2. Generate 3 high-level approaches to create tasks that conform to the model.
    3. For each approach, create 3 potential outcomes (best-case, average-case, worst-case).
    4. Evaluate the outcomes and select the most promising approach.
    5. Generate a sequence of {steps} tasks that conform to the model structure, based on the selected approach.

    The model structure is:
    {input_model.model_json_schema()}

    Return the tasks as a Plan object with a list of Task objects.
    """


if __name__ == "__main__":
    # Example usage with string input
    goal = "Create a marketing strategy for a new smartphone"
    plan_result = planner(goal, steps=5, verbose=True)
    print("Plan for string input:")
    for task in plan_result.tasks:
        print(f"- {task.description}")

    print("\n" + "=" * 50 + "\n")

    # Example usage with Pydantic model input
    from pydantic import BaseModel
    from typing import List

    class ResearchTask(BaseModel):
        topic: str
        resources: List[str]
        estimated_time: int

    plan_model_result = planner(ResearchTask, n=1, steps=4, verbose=True)
    print("Plan for Pydantic model input:")
    for task in plan_model_result.tasks:
        print(f"- Topic: {task.topic}")
        print(f"  Resources: {', '.join(task.resources)}")
        print(f"  Estimated Time: {task.estimated_time}")
        print()

    # Batch processing example
    batch_results = planner(
        ResearchTask, n=2, process="batch", batch_size=2, steps=3, verbose=True
    )
    print("Batch plans for Pydantic model input:")
    for i, plan in enumerate(batch_results, 1):
        print(f"Plan {i}:")
        for task in plan.tasks:
            print(f"- Topic: {task.topic}")
            print(f"  Resources: {', '.join(task.resources)}")
            print(f"  Estimated Time: {task.estimated_time}")
        print()