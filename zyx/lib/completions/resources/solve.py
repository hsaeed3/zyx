from typing import Callable, Optional, List, Any, Union, Literal
import time
import traceback
from loguru import logger

from ...types import client as clienttypes


def solve(
    prompt: str,
    model: str = "gpt-4o-mini",
    client: Literal["openai", "litellm"] = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: clienttypes.InstructorMode = "chat",
    max_retries: int = 3,
    run_tools: Optional[bool] = True,
    tools: Optional[List[clienttypes.ToolType]] = None,
    parallel_tool_calls: Optional[bool] = False,
    tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    verbose: Optional[bool] = False,
    num_samples: int = 10,
    verifier_func: Optional[Callable[[str], bool]] = None,
    **kwargs,
) -> Any:
    """
    Generates multiple solutions to the prompt using repeated sampling,
    and selects the correct one using the verifier function.

    Example:
        ```python
        def custom_verifier(sample: str) -> bool:
            # Implement domain-specific verification logic
            return "4" in sample  # Checks if '4' is in the sample

        tools = [add]  # List your tool functions here

        result = solve(
            prompt=prompt,
            num_samples=20,
            verifier_func=custom_verifier,
            verbose=True,
            run_tools=True,
            tools=tools,  # Pass the tools to be utilized
        )

        print(f"Final result: {result}")
    ```

    Args:
        prompt: The input prompt string.
        num_samples: Number of samples to generate.
        verifier_func: Function to verify if a sample is correct.
        return: The correct solution if found, else None.

    Returns:
        The correct solution if found, else None.
    """

    from ..client import completion

    if verifier_func is None:
        # Default verifier function
        def verifier_func(sample: str) -> bool:
            # Placeholder verification logic
            return "correct answer" in sample.lower()

    correct_samples = []
    for attempt in range(1, num_samples + 1):
        if verbose:
            print(f"Generating sample {attempt}/{num_samples}")
        try:
            response = completion(
                messages=prompt,
                model=model,
                client=client,
                api_key=api_key,
                base_url=base_url,
                mode=mode,
                max_retries=max_retries,
                run_tools=run_tools,
                tools=tools,
                parallel_tool_calls=parallel_tool_calls,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=stream,
                verbose=verbose,
                **kwargs,
            )

            # Extract the generated content
            sample = response.choices[0].message['content']

            if verbose:
                print(f"Sample {attempt}: {sample}")

            # Use the verifier function to check correctness
            if verifier_func(sample):
                correct_samples.append(sample)
                if verbose:
                    print(f"Found correct sample at attempt {attempt}")
                break  # Stop at first correct sample, or remove this line to collect all correct samples

            time.sleep(0.1)  # Optional delay to comply with API rate limits

        except Exception as e:
            logger.error(f"Error during sample generation: {e}")
            if verbose:
                traceback.print_exc()
            continue  # Continue to next attempt

    if correct_samples:
        return correct_samples[0]  # Return the first correct sample found
    else:
        if verbose:
            print("No correct sample found.")
        return None  # Or handle as appropriate for your application

# Example usage
if __name__ == "__main__":
    # Initialize parameters
    prompt = "Solve the following problem: What is 2 + 2?"

    # Define a custom verifier function if needed
    def custom_verifier(sample: str) -> bool:
        # Implement domain-specific verification logic
        return "4" in sample  # Checks if '4' is in the sample

    # Define tools if needed
    def add(a: int, b: int) -> int:
        """A tool function that adds two numbers."""
        return a + b

    tools = [add]  # List your tool functions here

    # Call the solve function
    result = solve(
        prompt=prompt,
        num_samples=20,
        verifier_func=custom_verifier,
        verbose=True,
        run_tools=True,
        tools=tools,  # Pass the tools to be utilized
    )

    print(f"Final result: {result}")
