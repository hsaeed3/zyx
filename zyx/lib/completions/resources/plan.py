from pydantic import BaseModel, create_model
from typing import Optional, List, Union, Literal
from ..client import Client
from ...types import client as clienttypes


class Task(BaseModel):
    description: str
    details: Optional[str] = None


class Plan(BaseModel):
    tasks: List[Task]


class BasePlan(BaseModel):
    goal: str
    approach: str


def plan(
    input: Union[str, BaseModel],
    process: Literal["least_to_most", "tree_of_thought"] = "least_to_most",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0,
    mode: clienttypes.InstructorMode = "markdown_json_mode",
    verbose: bool = False,
    steps: int = 5,
) -> Plan:
    """
    Generates a plan based on the input, using either the Least-to-Most or Tree of Thoughts prompting
    methods.

    Example:
        ```python
        import zyx
        from pydantic import BaseModel

        class ProjectTask(BaseModel):
            name: str
            duration: int
            dependencies: List[str]

        # Using a string input with Least-to-Most method
        plan1 = zyx.plan("Create a website for a small business", process="least_to_most", steps=7)

        # Using a Pydantic model input with Tree of Thoughts method
        plan2 = zyx.plan(ProjectTask, process="tree_of_thought", steps=3)
        ```

    Parameters:
        input (Union[str, BaseModel]): The input for plan generation, either a string or a Pydantic model.
        model (str): The model to use for generation.
        api_key (Optional[str]): The API key to use for generation.
        base_url (Optional[str]): The base URL to use for generation.
        organization (Optional[str]): The organization to use for generation.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries to make.
        temperature (float): The temperature to use for generation.
        mode (Literal["json", "md_json", "tools"]): The mode to use for generation.
        verbose (bool): Whether to print verbose output.
        process (Literal["least_to_most", "tree_of_thought"]): The planning method to use.
        steps (int): The number of steps to include in the plan (default is 5).

    Returns:
        Plan: A Plan object containing a list of generated tasks.
    """

    client = Client(client = "litellm")

    if process == "least_to_most":
        if isinstance(input, str):
            return _generate_plan_from_string_l2m(
                input,
                client,
                model,
                api_key,
                base_url,
                organization,
                max_tokens,
                max_retries,
                temperature,
                mode,
                verbose,
                steps,
            )
        elif isinstance(input, type) and issubclass(input, BaseModel):
            return _generate_plan_from_model_l2m(
                input,
                client,
                model,
                api_key,
                base_url,
                organization,
                max_tokens,
                max_retries,
                temperature,
                mode,
                verbose,
                steps,
            )
    elif process == "tree_of_thought":
        if isinstance(input, str):
            return _generate_plan_from_string_tot(
                input,
                client,
                model,
                api_key,
                base_url,
                organization,
                max_tokens,
                max_retries,
                temperature,
                mode,
                verbose,
                steps,
            )
        elif isinstance(input, type) and issubclass(input, BaseModel):
            return _generate_plan_from_model_tot(
                input,
                client,
                model,
                api_key,
                base_url,
                organization,
                max_tokens,
                max_retries,
                temperature,
                mode,
                verbose,
                steps,
            )
    else:
        raise ValueError(
            "Invalid planning method. Choose either 'least_to_most' or 'tree_of_thought'."
        )


# Update the function signatures and implementations for all four planning functions
def _generate_plan_from_string_l2m(
    input: str,
    client: Client,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    organization: Optional[str],
    max_tokens: Optional[int],
    max_retries: int,
    temperature: float,
    mode: str,
    verbose: bool,
    steps: int,
) -> Plan:
    system_message = f"""
    You are a planning assistant using the Least-to-Most prompting method. Your task is to generate a detailed plan based on the given input.
    Follow these steps:
    1. Identify the main goal or objective.
    2. Break down the goal into {steps} high-level steps.
    3. For each high-level step, provide 1-2 detailed sub-tasks.
    4. Ensure that each task is clear, actionable, and builds upon the previous ones.
    """

    # Step 1: Identify the main goal
    try:
        goal_response = _get_completion(
            client,
            model,
            api_key,
            base_url,
            organization,
            max_tokens,
            max_retries,
            temperature,
            mode,
            verbose,
            system_message,
            f"What is the main goal or objective for this plan: {input}",
        )
    except Exception as e:
        if verbose:
            print(f"Error generating goal: {e}")

    if verbose:
        print(f"Goal: {goal_response}")

    # Step 2: Break down into high-level steps
    try:
        high_level_steps_response = _get_completion(
            client,
            model,
            api_key,
            base_url,
            organization,
            max_tokens,
            max_retries,
            temperature,
            mode,
            verbose,
            system_message,
            f"Break down the goal '{goal_response}' into {steps} high-level steps.",
        )
    except Exception as e:
        if verbose:
            print(f"Error generating high-level steps: {e}")

    if verbose:
        print(f"Steps Created: {len(high_level_steps_response)}")

    # Step 3 & 4: Provide detailed sub-tasks for each high-level step
    try:
        detailed_tasks = []
        for step in high_level_steps_response.split("\n"):
            sub_tasks_response = _get_completion(
                client,
                model,
                api_key,
                base_url,
                organization,
                max_tokens,
                max_retries,
                temperature,
                mode,
                verbose,
                system_message,
                f"Provide 1-2 detailed, actionable sub-tasks for the step: {step}",
            )
            detailed_tasks.extend(
                [
                    Task(description=sub_task.strip())
                    for sub_task in sub_tasks_response.split("\n")
                    if sub_task.strip()
                ]
            )
    except Exception as e:
        if verbose:
            print(f"Error generating detailed tasks: {e}")

    return Plan(tasks=detailed_tasks)


def _generate_plan_from_model_l2m(
    input_model: type,
    client: Client,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    organization: Optional[str],
    max_tokens: Optional[int],
    max_retries: int,
    temperature: float,
    mode: str,
    verbose: bool,
    steps: int,
) -> Plan:
    system_message = f"""
    You are a planning assistant using the Least-to-Most prompting method. Your task is to generate a detailed plan based on the given Pydantic model.
    Follow these steps:
    1. Analyze the provided model structure.
    2. Generate a sequence of {steps} tasks that conform to the model.
    3. Ensure each task builds upon the previous ones and follows the model's constraints.

    The model structure is:
    {input_model.model_json_schema()}

    Return the tasks as a list of Python dictionaries, with each dictionary representing a single task.
    """

    # Step 1: Analyze the model structure
    try:
        model_analysis = _get_completion(
            client,
            model,
            api_key,
            base_url,
            organization,
            max_tokens,
            max_retries,
            temperature,
            mode,
            verbose,
            system_message,
            "Analyze the provided model structure and explain its key components.",
        )
    except Exception as e:
        if verbose:
            print(f"Error generating model analysis: {e}")

    if verbose:
        print(f"Model Analysis: {model_analysis}")

    # Step 2 & 3: Generate sequence of tasks
    try:
        tasks_response = _get_completion(
            client,
            model,
            api_key,
            base_url,
            organization,
            max_tokens,
            max_retries,
            temperature,
            mode,
            verbose,
            system_message,
            f"Based on the model analysis: {model_analysis}\nGenerate a sequence of {steps} tasks that conform to the model structure. Ensure each task builds upon the previous ones. Return the tasks as a list of Python dictionaries.",
        )
    except Exception as e:
        if verbose:
            print(f"Error generating tasks: {e}")

    if verbose:
        print(f"Tasks Response: {tasks_response}")

    # Parse the response and create Task objects
    tasks = []
    try:
        task_list = eval(
            tasks_response
        )  # Assuming the response is a list of dictionaries
        if not isinstance(task_list, list):
            raise ValueError("Expected a list of tasks")

        for task_dict in task_list:
            task_instance = input_model(**task_dict)
            tasks.append(
                Task(
                    description=str(task_instance),
                    details=task_instance.model_dump_json(),
                )
            )
    except Exception as e:
        if verbose:
            print(f"Error parsing tasks: {tasks_response}")
            print(f"Error message: {str(e)}")

    if not tasks:
        raise ValueError("Failed to generate valid tasks from the model input.")

    return Plan(tasks=tasks)


def _generate_plan_from_string_tot(
    input: str,
    client: Client,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    organization: Optional[str],
    max_tokens: Optional[int],
    max_retries: int,
    temperature: float,
    mode: str,
    verbose: bool,
    steps: int,
) -> Plan:
    system_message = f"""
    You are a planning assistant using the Tree of Thoughts method. Your task is to generate a detailed plan based on the given input.
    Follow these steps:
    1. Generate 3 high-level approaches to achieve the goal.
    2. For each approach, create 3 potential outcomes (best-case, average-case, worst-case).
    3. Evaluate the outcomes and select the most promising approach.
    4. Break down the selected approach into {steps} detailed, actionable tasks.
    """

    # Step 1: Generate 3 high-level approaches
    try:
        approaches_response = _get_completion(
            client,
            model,
            api_key,
            base_url,
            organization,
            max_tokens,
            max_retries,
            temperature,
            mode,
            verbose,
            system_message,
            f"Generate 3 high-level approaches to achieve this goal: {input}",
        )
    except Exception as e:
        if verbose:
            print(f"Error generating approaches: {e}")

    if verbose:
        print(f"Approaches: {approaches_response}")

    # Step 2 & 3: Generate outcomes and evaluate
    try:
        evaluation_response = _get_completion(
            client,
            model,
            api_key,
            base_url,
            organization,
            max_tokens,
            max_retries,
            temperature,
            mode,
            verbose,
            system_message,
            f"For each approach:\n{approaches_response}\n\nGenerate 3 potential outcomes (best-case, average-case, worst-case) and evaluate them. Then, select the most promising approach.",
        )
    except Exception as e:
        if verbose:
            print(f"Error generating evaluation: {e}")

    if verbose:
        print(f"Evaluation: {evaluation_response}")

    # Step 4: Break down the selected approach
    try:
        tasks_response = _get_completion(
            client,
            model,
            api_key,
            base_url,
            organization,
            max_tokens,
            max_retries,
            temperature,
            mode,
            verbose,
            system_message,
            f"Based on the evaluation:\n{evaluation_response}\n\nBreak down the selected approach into {steps} detailed, actionable tasks.",
        )
    except Exception as e:
        if verbose:
            print(f"Error generating tasks: {e}")

    if verbose:
        print(f"Tasks Response: {tasks_response}")

    tasks = [
        Task(description=task.strip())
        for task in tasks_response.split("\n")
        if task.strip()
    ]
    return Plan(tasks=tasks)


def _generate_plan_from_model_tot(
    input_model: type,
    client: Client,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    organization: Optional[str],
    max_tokens: Optional[int],
    max_retries: int,
    temperature: float,
    mode: str,
    verbose: bool,
    steps: int,
) -> Plan:
    system_message = f"""
    You are a planning assistant using the Tree of Thoughts method. Your task is to generate a detailed plan based on the given Pydantic model.
    Follow these steps:
    1. Analyze the provided model structure.
    2. Generate 3 high-level approaches to create tasks that conform to the model.
    3. For each approach, create 3 potential outcomes (best-case, average-case, worst-case).
    4. Evaluate the outcomes and select the most promising approach.
    5. Generate a sequence of {steps} tasks that conform to the model structure, based on the selected approach.

    The model structure is:
    {input_model.model_json_schema()}

    Return the final tasks as a list of Python dictionaries, with each dictionary representing a single task.
    """

    # Steps 1-4: Analyze, generate approaches, evaluate, and select
    try:
        analysis_response = _get_completion(
            client,
            model,
            api_key,
            base_url,
            organization,
            max_tokens,
            max_retries,
            temperature,
            mode,
            verbose,
            system_message,
            "Perform steps 1-4 as described in the instructions.",
        )
    except Exception as e:
        if verbose:
            print(f"Error generating analysis: {e}")

    if verbose:
        print(f"Analysis and Approach Selection: {analysis_response}")

    # Step 5: Generate tasks based on the selected approach
    try:
        tasks_response = _get_completion(
            client,
            model,
            api_key,
            base_url,
            organization,
            max_tokens,
            max_retries,
            temperature,
            mode,
            verbose,
            system_message,
            f"Based on the analysis and selected approach:\n{analysis_response}\n\nGenerate a sequence of {steps} tasks that conform to the model structure. Return the tasks as a list of Python dictionaries.",
        )
    except Exception as e:
        if verbose:
            print(f"Error generating tasks: {e}")

    if verbose:
        print(f"Tasks Response: {tasks_response}")

    # Parse the response and create Task objects
    tasks = []
    try:
        task_list = eval(
            tasks_response
        )  # Assuming the response is a list of dictionaries
        if not isinstance(task_list, list):
            raise ValueError("Expected a list of tasks")

        for task_dict in task_list:
            task_instance = input_model(**task_dict)
            tasks.append(
                Task(
                    description=str(task_instance),
                    details=task_instance.model_dump_json(),
                )
            )
    except Exception as e:
        if verbose:
            print(f"Error parsing tasks: {tasks_response}")
            print(f"Error message: {str(e)}")

    if not tasks:
        raise ValueError("Failed to generate valid tasks from the model input.")

    return Plan(tasks=tasks)


# Existing helper function
def _get_completion(
    client: Client,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    organization: Optional[str],
    max_tokens: Optional[int],
    max_retries: int,
    temperature: float,
    mode: str,
    verbose: bool,
    system_message: str,
    user_message: str,
) -> str:
    ResponseModel = create_model("ResponseModel", content=(str, ...))

    try:
        response = client.completion(
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
            mode="md_json" if model.startswith(("ollama/", "ollama_chat/")) else mode,
            response_model=ResponseModel,
        )
    except Exception as e:
        if verbose:
            print(f"Error generating completion: {e}")

    return response.content


if __name__ == "__main__":
    from pydantic import BaseModel
    from typing import List

    # Test with string input using Least-to-Most method
    print("Testing with string input (Least-to-Most):")
    plan_string_l2m = plan(
        "Create a website for a small business",
        verbose=True,
        process="least_to_most",
        steps=7,
    )
    print(plan_string_l2m)
    print("\n" + "=" * 50 + "\n")

    # Test with string input using Tree of Thoughts method
    print("Testing with string input (Tree of Thoughts):")
    plan_string_tot = plan(
        "Create a website for a small business",
        verbose=True,
        process="tree_of_thought",
        steps=3,
    )
    print(plan_string_tot)
    print("\n" + "=" * 50 + "\n")

    # Test with BaseModel input using Least-to-Most method
    print("Testing with BaseModel input (Least-to-Most):")

    class ProjectTask(BaseModel):
        name: str
        duration: int
        dependencies: List[str]

    plan_model_l2m = plan(ProjectTask, verbose=True, process="least_to_most", steps=4)
    print(plan_model_l2m)
    print("\n" + "=" * 50 + "\n")

    # Test with BaseModel input using Tree of Thoughts method
    print("Testing with BaseModel input (Tree of Thoughts):")
    plan_model_tot = plan(ProjectTask, verbose=True, process="tree_of_thought", steps=6)
    print(plan_model_tot)
    print("\n" + "=" * 50 + "\n")