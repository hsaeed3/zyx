from ...types import client as clienttypes
from pydantic import BaseModel, Field
from typing import Callable, Optional, Literal, get_type_hints, Any
import traceback


class FunctionResponse(BaseModel):
    code: str
    output: Any


def function(
    model: str = "gpt-4o-mini",
    client: Literal["litellm", "openai"] = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: clienttypes.InstructorMode = "markdown_json_mode",
    mock: bool = False,
    return_code : bool = False,
    verbose: bool = False,
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
        from .... import completion
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
            retry=tenacity.retry_if_exception_type((RuntimeError, Exception)),
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
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

                error_context = ""
                try:
                    # Prepare the import statement and return type
                    import_statement = "import typing\n"
                    return_type_str = (
                        f"typing.Any" if return_type == Any else str(return_type)
                    )

                    messages = [
                        {
                            "role": "system",
                            "content": f"""
                            ## CONTEXT ##
                            
                            You are a Python code generator. Your goal is to generate a Python function that matches this specification:
                            Function: {f.__name__}
                            Arguments and their types: {function_args}
                            Return type: {return_type_str}
                            Description: {f.__doc__} \n
                            
                            ## OBJECTIVE ##
                            
                            Generate the complete Python code as a single string, including all necessary import statements.
                            The code should define the function and include a call to the function with the provided inputs.
                            The last line should assign the result of the function call to a variable named 'result'.
                            Do not include any JSON encoding, printing, or explanations in your response.
                            Ensure all required modules are imported.
                            {error_context}
                            """,
                        },
                        {
                            "role": "user",
                            "content": f"Generate code for the function with these inputs: {input_dict}",
                        },
                    ]
                    response = completion(
                        client=client,
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

                    if verbose:
                        print(f"Code generation response: {response}")

                    # Prepend the import statement to the generated code
                    full_code = import_statement + response.code

                    # Create a temporary Python file
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".py", delete=False
                    ) as temp_file:
                        temp_file.write(full_code)

                    # Execute the temporary Python file
                    env = os.environ.copy()
                    env["PYTHONPATH"] = ":".join(sys.path)
                    result = subprocess.run(
                        [sys.executable, temp_file.name],
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    from rich.prompt import Confirm

                    if result.returncode != 0:
                        if "ModuleNotFoundError" in result.stderr:
                            missing_library = result.stderr.split("'")[1]
                            install = Confirm.ask(
                                prompt=f"The library '[bold blue3]{missing_library}[/bold blue3]' is required but not installed. Do you want to install it?"
                            )
                            if install:
                                subprocess.check_call(
                                    [
                                        sys.executable,
                                        "-m",
                                        "pip",
                                        "install",
                                        missing_library,
                                    ]
                                )
                                print(
                                    f"Installed {missing_library}. Rerunning the code."
                                )
                                # Rerun the code after installation
                                result = subprocess.run(
                                    [sys.executable, temp_file.name],
                                    capture_output=True,
                                    text=True,
                                    env=env,
                                )
                                if result.returncode != 0:
                                    raise RuntimeError(
                                        f"Error executing generated code after library installation: {result.stderr}"
                                    )
                            else:
                                raise RuntimeError(
                                    f"Required library '{missing_library}' is not installed. Please install it manually and try again."
                                )
                        else:
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
                    if return_code:
                        return FunctionResponse(code=full_code, output=module.result)
                    return module.result

                except Exception as e:
                    print(f"Error in code generation or execution, retrying...")

                    error_context = f"""
                    Previous attempt failed with the following error:
                    {str(e)}
                    
                    Traceback:
                    {traceback.format_exc()}
                    
                    Please adjust the code to avoid this error and ensure it runs successfully.
                    """
                    raise RuntimeError(
                        f"Error in code generation or execution: {str(e)}"
                    )

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
                response = completion(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    response_model=FunctionResponseModel,
                    mode = "markdown_json_mode"
                    if model.startswith(("ollama/", "ollama_chat/"))
                    else mode,
                    **kwargs,
                )
                if return_code:
                    return FunctionResponse(code="", output=response.output)
                return response.output

        return wrapper

    return decorator


if __name__ == "__main__":

    @function(return_code=True)
    def add(a: int, b: int) -> int:
        """
        Add two numbers.
        """
        return a + b

    print(add(1, 2))