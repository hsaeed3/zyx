__all__ = ["function"]

from typing import Callable, Optional, Literal, get_type_hints, Any
from ...core.main import BaseModel, Field


def function(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: Literal["json", "md_json", "tools"] = "md_json",
    mock: bool = True,
    **kwargs,
) -> Callable[[Callable], Callable]:
    """A quick abstraction to create both mock & generated runnable python functions, using LLMs.

    Parameters:
        model (str): The model to use for the function.
        api_key (Optional[str]): The API key to use for the function.
        base_url (Optional[str]): The base URL to use for the function.
        mode (Literal["json", "md_json", "tools"]): The mode to use for the function.
        mock (bool): Whether to use the mock mode for the function.
        kwargs (dict): Additional keyword arguments to pass to the function.

    Returns:
        The function response or the generated code response (Any).
    """

    def decorator(f: Callable) -> Callable:
        from ..main import Client
        from pydantic import create_model
        from functools import wraps
        import tenacity
        import tempfile
        import subprocess
        import sys
        import json
        import os
        import ast
        import importlib

        @wraps(f)
        @tenacity.retry(
            stop=tenacity.stop_after_attempt(3),
            retry=tenacity.retry_if_exception_type(RuntimeError),
        )
        def wrapper(*args, **kwargs):
            type_hints = get_type_hints(f)
            return_type = type_hints.pop("return", Any)
            function_args = {k: v for k, v in type_hints.items()}
            input_dict = dict(zip(function_args.keys(), args))
            input_dict.update(kwargs)

            if not mock:

                class CodeGenerationModel(BaseModel):
                    code: str = Field(
                        ..., description="Complete Python code as a single string"
                    )

                messages = [
                    {
                        "role": "system",
                        "content": f"""
                        ## CONTEXT ##
                        
                        You are a Python code generator. Your goal is to generate a Python function that matches this specification:
                        Function: {f.__name__}
                        Arguments and their types: {function_args}
                        Return type: {return_type}
                        Description: {f.__doc__} \n
                        
                        ## OBJECTIVE ##
                        
                        Generate the complete Python code as a single string, including all necessary import statements.
                        The code should define the function and include a call to the function with the provided inputs.
                        The last line should assign the result of the function call to a variable named 'result'.
                        Do not include any JSON encoding, printing, or explanations in your response.
                        Ensure all required modules are imported.
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"Generate code for the function with these inputs: {input_dict}",
                    },
                ]
                response = Client().completion(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    response_model=CodeGenerationModel,
                    mode="md_json"
                    if model.startswith(("ollama/", "ollama_chat/"))
                    else mode,
                    **kwargs,
                )

                # Create a temporary Python file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as temp_file:
                    temp_file.write(response.code)

                # Execute the temporary Python file
                env = os.environ.copy()
                env["PYTHONPATH"] = ":".join(sys.path)
                result = subprocess.run(
                    [sys.executable, temp_file.name],
                    capture_output=True,
                    text=True,
                    env=env,
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Error executing generated code: {result.stderr}"
                    )

                # Import the generated module
                spec = importlib.util.spec_from_file_location(
                    "generated_module", temp_file.name
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Return the result directly
                return module.result

            else:
                FunctionResponseModel = create_model(
                    "FunctionResponseModel",
                    output=(return_type, ...),
                )
                messages = [
                    {
                        "role": "system",
                        "content": f"""
                        You are a Python function emulator. Your goal is to simulate the response of this Python function:
                        Function: {f.__name__}
                        Arguments and their types: {function_args}
                        Return type: {return_type}
                        Description: {f.__doc__}
                        Respond only with the output the function would produce, without any additional explanation.
                        """,
                    },
                    {"role": "user", "content": f"Function inputs: {input_dict}"},
                ]
                response = Client().completion(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    response_model=FunctionResponseModel,
                    mode="md_json"
                    if model.startswith(("ollama/", "ollama_chat/"))
                    else mode,
                    **kwargs,
                )
                return response.output

        return wrapper

    return decorator


if __name__ == "__main__":

    @function()
    def add(a: int, b: int) -> int:
        """
        Add two numbers.
        """
        return a + b

    print(add(1, 2))
