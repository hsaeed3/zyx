from pydantic import BaseModel, create_model, Field
from typing import Optional, List, Union, Literal, Type, Any
from ....lib.utils.logger import get_logger
from ....client import Client, InstructorMode

logger = get_logger("plan")


class Task(BaseModel):
    description: str
    details: Optional[str] = None


class Plan(BaseModel):
    tasks: List[Task]


def plan(
    input: Union[str, Type[BaseModel]],
    instructions: Optional[str] = None,
    process: Literal["single", "batch"] = "single",
    n: int = 1,
    batch_size: int = 3,
    steps: int = 5,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    temperature: Optional[float] = None,
    mode: InstructorMode = "markdown_json_mode",
    max_retries: int = 3,
    client: Optional[Literal["openai", "litellm"]] = None,
    verbose: bool = False,
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
        provider=client,
        verbose=verbose,
    )

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
    plan_result = plan(goal, steps=5, verbose=True)
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

    plan_model_result = plan(ResearchTask, n=1, steps=4, verbose=True)
    print("Plan for Pydantic model input:")
    for task in plan_model_result.tasks:
        print(f"- Topic: {task.topic}")
        print(f"  Resources: {', '.join(task.resources)}")
        print(f"  Estimated Time: {task.estimated_time}")
        print()

    # Batch processing example
    batch_results = plan(
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
