from .. import _rich as utils
from ..resources.types import completion_create_params as params
from .. import _Client as Client


from pydantic import BaseModel, Field
from typing import Any, Literal, Optional, Union
import tenacity
import traceback
import tempfile
import sys
import os
from rich.progress import Progress, SpinnerColumn, TextColumn
import re


def coder(
    description: str,
    model: Union[str, params.ChatModel] = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    mode: params.InstructorMode = "tool_call",
    temperature: Optional[float] = None,
    provider: Optional[Literal["openai", "litellm"]] = "openai",
    progress_bar: Optional[bool] = True,
    client: Optional[Client] = None,
    return_code: bool = False,
    verbose: bool = False,
    max_retries: int = 3,
    **kwargs,
) -> Any:
    """
    Generates, executes and returns results of python code.

    Args:
        // ... existing args ...
        max_retries (int): Maximum number of retries if code generation/execution fails
    """
    
    last_error = None
    last_code = None

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(max_retries),
        retry=tenacity.retry_if_exception_type((RuntimeError, Exception)),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=lambda retry_state: _handle_retry(retry_state, description),
    )
    def _generate_and_execute():
        nonlocal last_error, last_code
        
        if verbose:
            utils.logger.info(f"Generating code based on description: {description}")
            utils.logger.info(f"Using model: {model}")

        class CodeGenerationModel(BaseModel):
            code: str = Field(..., description="Complete Python code as a single string")

        base_instructions = """
        You are a Python code generator. Your goal is to generate Python code based on the given description.

        # Instructions

        - Generate the complete Python code as a single string.
        - The code should define any necessary functions or classes.
        - Do not include any JSON encoding, printing, or explanations in your response.
        - All arguments in the code should be typed.
        - The function must include a clear docstring explaining its purpose, arguments and return value.
        - All imports must be nested inside the function body, not at the module level.
        - Do not include any JSON encoding, printing, or explanations in your response.
        """

        if return_code:
            system_message = base_instructions + "\n  - The last line MUST be the return statement of the function."
            user_message = f"Generate Python code for: {description}"
        else:
            system_message = base_instructions + "\n    - The last line MUST assign the created object to a variable named 'result'."
            user_message = f"Generate Python code to create this object and assign it to 'result': {description}"

        completion_client = Client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            provider=provider,
            verbose=verbose,
        ) if client is None else client

        try:
            if progress_bar:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True
                ) as progress:
                    task_id = progress.add_task("Constructing...", total=None)

                    response = completion_client.completion(
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                        model=model,
                        response_model=CodeGenerationModel,
                        mode=mode,
                        temperature=temperature,
                        progress_bar=False,
                        **kwargs,
                    )

                    progress.update(task_id, completed=1)
            else:
                response = completion_client.completion(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    model=model,
                    response_model=CodeGenerationModel,
                    mode=mode,
                    temperature=temperature,
                    progress_bar=False,
                    **kwargs,
                )

            if verbose:
                utils.logger.info(f"Generated code:\n{response.code}")

            # Ensure the code assigns to 'result'
            if "result =" not in response.code:
                response.code += "\nresult = " + description.split()[-1]

            # Create a temporary Python file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(response.code)
                temp_file_path = temp_file.name

            # Add the current directory to sys.path to allow importing the temporary module
            sys.path.insert(0, os.path.dirname(temp_file_path))
            try:
                if return_code:
                    # Clean up any result assignment at the end if return_code=True
                    cleaned_code = re.sub(r'\nresult\s*=\s*.+$', '', response.code)
                    return cleaned_code
                    
                # Execute the generated code in a local namespace
                local_namespace = {}
                exec(response.code, {}, local_namespace)

                # Return the result object
                if "result" not in local_namespace:
                    raise ValueError("No result object found in the generated code.")

                return local_namespace["result"]

            finally:
                # Clean up: remove the temporary file
                os.unlink(temp_file_path)
                sys.path.pop(0)

        except Exception as e:
            last_error = str(e)
            last_code = response.code if 'response' in locals() else None
            raise RuntimeError(f"Error in code generation or execution: {str(e)}")

    def _handle_retry(retry_state, description):
        """Handle retry by updating the prompt with error context"""
        nonlocal last_error, last_code
        
        if last_code and last_error:
            # Update description with context from the last failure
            return f"""Previous attempt failed with error: {last_error}
            
            Previous code that failed:
            {last_code}
            
            Please fix the code to handle this error and try again.
            Do not change the original request. Do not add any other instructions.
            You should generate the full code.
            Original request: {description}"""
        return description

    return _generate_and_execute()


if __name__ == "__main__":
    # Generate a logger object
    generated_logger = coder(
        "create a logger named 'my_logger' that logs to console with INFO level",
        verbose=True,
    )

    # Use the generated logger
    generated_logger.info("This is a test log message")
