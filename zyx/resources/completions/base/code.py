from ....lib.utils.logger import get_logger
from ....client import Client, InstructorMode
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
import traceback
import tempfile
import sys
import os


logger = get_logger("code")


def code(
    description: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    mode: InstructorMode = "tool_call",
    temperature: Optional[float] = None,
    client: Optional[Literal["openai", "litellm"]] = None,
    verbose: bool = False,
    **kwargs,
) -> Any:
    """
    Generates, executes and returns results of python code.
    """

    if verbose:
        logger.info(f"Generating code based on description: {description}")
        logger.info(f"Using model: {model}")

    class CodeGenerationModel(BaseModel):
        code: str = Field(..., description="Complete Python code as a single string")

    system_message = """
    You are a Python code generator. Your goal is to generate Python code based on the given description.
    Generate the complete Python code as a single string, including all necessary import statements.
    The code should define any necessary functions or classes and create the described object.
    The last line MUST assign the created object to a variable named 'result'.
    Do not include any JSON encoding, printing, or explanations in your response.
    Ensure all required modules are imported.
    """

    user_message = f"Generate Python code to create this object and assign it to 'result': {description}"

    completion_client = Client(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        provider=client,
        verbose=verbose,
    )

    try:
        response = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=model,
            response_model=CodeGenerationModel,
            mode=mode,
            temperature=temperature,
            **kwargs,
        )

        if verbose:
            logger.info(f"Generated code:\n{response.code}")

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
        print(f"Error in code generation or execution: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    # Generate a logger object
    generated_logger = code(
        "create a logger named 'my_logger' that logs to console with INFO level",
        verbose=True,
    )

    # Use the generated logger
    generated_logger.info("This is a test log message")
