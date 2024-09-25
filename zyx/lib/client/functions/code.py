from ...types import client as clienttypes
from pydantic import BaseModel, Field
from typing import Optional, Literal, Any


def code(
    description: str,
    client: Literal["litellm", "openai"] = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: clienttypes.InstructorMode = "markdown_json_mode",
    verbose: bool = False,
    **kwargs,
) -> Any:
    """Generate, execute Python code based on a string description, and return the resulting object.

    Example:
        ```python
        import zyx

        logger = zyx.llm.code("create a logger named 'my_logger' that logs to console with INFO level")
        logger.info("This is a test log message")
        ```

    Parameters:
        description (str): A string describing the desired object or functionality.
        model (str): The model to use for code generation.
        api_key (Optional[str]): The API key to use for the function.
        base_url (Optional[str]): The base URL to use for the function.
        mode (Literal["json", "md_json", "tools"]): The mode to use for the function.
        verbose (bool): Whether to print verbose output.
        kwargs (dict): Additional keyword arguments to pass to the function.

    Returns:
        The object created by executing the generated code.
    """
    from .... import completion
    import traceback
    import tempfile
    import sys
    import os
    import importlib

    class CodeGenerationModel(BaseModel):
        code: str = Field(..., description="Complete Python code as a single string")

    try:
        messages = [
            {
                "role": "system",
                "content": """
                You are a Python code generator. Your goal is to generate Python code based on the given description.
                Generate the complete Python code as a single string, including all necessary import statements.
                The code should define any necessary functions or classes and create the described object.
                The last line should assign the created object to a variable named 'result'.
                Do not include any JSON encoding, printing, or explanations in your response.
                Ensure all required modules are imported.
                """,
            },
            {
                "role": "user",
                "content": f"Generate Python code to create this object: {description}",
            },
        ]
        response = completion(
            client = client,
            messages=messages,
            model=model,
            api_key=api_key,
            base_url=base_url,
            response_model=CodeGenerationModel,
            mode="markdown_json_mode" if model.startswith(("ollama/", "ollama_chat/")) else mode,
            **kwargs,
        )

        if verbose:
            print(f"Generated code:\n{response.code}")

        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(response.code)
            temp_file_path = temp_file.name

        # Add the current directory to sys.path to allow importing the temporary module
        sys.path.insert(0, os.path.dirname(temp_file_path))

        try:
            # Import the generated module
            module_name = os.path.splitext(os.path.basename(temp_file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Return the result object
            return module.result
        finally:
            # Clean up: remove the temporary file and restore sys.path
            os.unlink(temp_file_path)
            sys.path.pop(0)

    except Exception as e:
        print(f"Error in code generation or execution: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


# Example usage
if __name__ == "__main__":
    # Generate a logger object
    logger = code(
        "create a logger named 'my_logger' that logs to console with INFO level"
    )

    # Use the generated logger
    logger.info("This is a test log message")